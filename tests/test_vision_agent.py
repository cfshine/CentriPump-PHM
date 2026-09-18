"""Step 3 唯一节点 ``vision_node``：契约、工具、OCR 判噪声、失败隔离、重试、真实图。

★ 这里**不再用「吞掉任何异常 → skip」**的写法：
  只有两个**前置条件**缺失才 skip（没有 DEEPSEEK_API_KEY / 没有样例图）；
  节点建不起来、或视觉调用失败，都必须**红**，而不是静悄悄地绿。

    pytest tests/test_vision_agent.py -q -rs     # -rs 会把 skip 原因打出来

2026-09-17 精简后：Step 3 不再是子图，而是一个普通节点函数 ``vision_node``；
结构化输出只保留 3 个字段（natural_description / observations / limitations）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

_HAS_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))
requires_key = pytest.mark.skipif(
    not _HAS_KEY, reason="需要 DEEPSEEK_API_KEY（写在 ~/.bashrc，用交互式终端跑 pytest）"
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample_inputs"


def _sample_images() -> list[Path]:
    """**递归**找样例图 —— 图片放在 data/sample_inputs/vision/ 里也能被找到。"""
    return sorted(
        p for p in _SAMPLE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _as_findings(value) -> VisionFindings:
    """节点返回的可能是模型实例、也可能是 dict，统一成 VisionFindings。"""
    return value if isinstance(value, VisionFindings) else VisionFindings.model_validate(value)


from src.schemas.state import DiagnosisState
from src.schemas.vision import Observation, VisionFindings, VisionImage
from src.sub_agents.vision_agent import vision_client as vc
from src.sub_agents.vision_agent import vision_nodes as vn
from src.sub_agents.vision_agent import vision_pipeline as vp
from src.sub_agents.vision_agent.image_io import detect_content_type, to_data_url
from src.sub_agents.vision_agent.vision_compose import FAILED_PREFIX, compose_description
from src.sub_agents.vision_agent.ocr_quality import assess_ocr, evaluate_ocr_quality
from src.sub_agents.vision_agent.ocr_runner import OcrResult

# 1×1 红 PNG / 极小 JPEG：只用来验证「按魔数判 MIME」，不依赖 Pillow。
# ★ PNG 这串必须是**真能解码**的（旧版本少了一段合法的 IDAT，装上 Pillow 后
#   一解码就 OSError: broken data stream —— 现在 resize_for_api 同样会解码）。
_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415478da63f8cfc0000003010100f70341430000000049454e44ae426082"
)
_MIN_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00" + bytes([8] * 64)
    + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f\xff\xd9"
)


# =============================================================================
# 1. 契约：公共字段 + 盒子只有 3 个字段
# =============================================================================


def test_public_contract_shares_description_and_typed_box() -> None:
    """公共契约 = 人读的一段的描述 + 机器读的强类型盒子；被剔除的字段不许回来。"""
    public = set(DiagnosisState.model_fields)
    assert {"visual_description", "visual_findings"} <= public
    assert DiagnosisState.model_fields["visual_findings"].annotation is VisionFindings, (
        "visual_findings 必须是强类型 VisionFindings，不是 dict[str, Any] 袋子"
    )
    assert not ({"visual_rag_queries", "content_type", "extracted_text",
                 "route_used"} & public)


def test_box_has_exactly_three_fields_per_image() -> None:
    """★ 契约精简：每张图**只有 3 个字段**，多一个都不许回来（防回退回潮）。"""
    assert set(VisionImage.model_fields) == {
        "natural_description", "observations", "limitations"
    }
    assert set(Observation.model_fields) == {"target", "finding", "polarity"}
    assert set(VisionFindings.model_fields) == {"images"}


# =============================================================================
# 2. 工具层（不需要网络）
# =============================================================================


def test_detect_content_type_png_and_jpeg() -> None:
    assert detect_content_type(_MIN_PNG) == "image/png"
    assert detect_content_type(_MIN_JPEG) == "image/jpeg"


def test_detect_content_type_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        detect_content_type(b"not-an-image")


def test_to_data_url_prefix() -> None:
    url = to_data_url(_MIN_PNG)
    assert url.startswith("data:image/png;base64,")
    assert to_data_url(_MIN_JPEG, "image/jpeg").startswith("data:image/jpeg;base64,")


def _png_bytes(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_resize_for_api_passes_through_when_within_limits() -> None:
    """★ 已在限制内的图**原样返回**，一次都不重新编码。

    实测（9 张样例图 9/9 命中）：PIL 的 `save(PNG, optimize=True)` 对本来不需要
    缩放的 1~3MB PNG 要花 0.7~4.6 秒，占整张图处理时间的 32% —— 纯白工。
    """
    from src.sub_agents.vision_agent.image_io import resize_for_api

    Image = pytest.importorskip("PIL.Image")
    small = _png_bytes(Image.new("RGB", (200, 100), (10, 20, 30)))

    assert resize_for_api(small) is small, "应该原字节返回（连重新编码都没有）"


def test_resize_for_api_still_shrinks_oversized() -> None:
    """超出边长上限的图仍然要缩（快速通道不能把该做的活也跳过）。"""
    import io

    from src.sub_agents.vision_agent.image_io import resize_for_api

    Image = pytest.importorskip("PIL.Image")
    big = _png_bytes(Image.new("RGB", (3000, 100), (10, 20, 30)))

    out = resize_for_api(big)
    assert out is not big, "超限 → 必须走重新编码"
    assert detect_content_type(out) == "image/png"
    assert max(Image.open(io.BytesIO(out)).size) <= 2048


# =============================================================================
# 3. OCR：只判"噪声 / 有用信息"
# =============================================================================


def test_ocr_quality_useful_vs_noise() -> None:
    """体检只回答一件事：这段文字是**有用信息**还是**噪声**。

    三条判据：字数够（≥24）、有数字（≥4）、属于工业语境（命中单位/位号/字段名之一）。
    """
    useful = evaluate_ocr_quality(
        chars=392, digits=81,
        units=("mpa",), tags=(), keys=("LOCATION", "MODEL", "STATUS", "TEMP"),
    )
    assert useful.credible, "真实 HMI 的文字是有用信息 → 当提示词喂给模型"

    assert evaluate_ocr_quality(chars=7, digits=0).credible is False, \
        "照片上的 'ww Wink' 是噪声 → 丢弃"
    assert evaluate_ocr_quality(chars=300, digits=30, units=(), tags=(), keys=()).credible is False, \
        "没有单位/位号/字段名 → 不属于工业语境，仍算噪声"
    assert evaluate_ocr_quality(chars=200, digits=0, units=("mpa",), tags=(), keys=()).credible is False, \
        "没有数字 → 不是读数类有用信息"


def test_ocr_quality_clean_text_screen_is_useful() -> None:
    """干净的数字屏幕：字多、有数字、有单位/位号 → 有用信息。"""
    verdict = evaluate_ocr_quality(
        chars=122, digits=20,
        units=("mpa", "m3/h", "mm/s"), tags=("FAL-104",),
        keys=("ALARM", "MODEL", "PRESS", "PV", "SP", "TAG", "TIME"),
    )
    assert verdict.credible


def test_ocr_quality_real_hmi_is_useful() -> None:
    """实测真实 HMI 截图：字够多、有单位/字段名 → 有用信息，喂给模型当参考。"""
    verdict = evaluate_ocr_quality(
        chars=392, digits=40,
        units=("mpa",), tags=(), keys=("LOCATION", "MODEL", "STATUS", "TEMP"),
    )
    assert verdict.credible
    assert verdict.units == ("mpa",)


def test_ocr_quality_rejects_photo_noise() -> None:
    """真实照片上的 OCR 噪声（实测泄漏照吐出 'ww Wink'）→ 噪声，不喂模型。"""
    assert evaluate_ocr_quality(chars=7, digits=0).credible is False


def test_assess_ocr_empty_text_needs_no_tesseract() -> None:
    verdict = assess_ocr(b"whatever", "")
    assert not verdict.credible
    assert (verdict.chars, verdict.digits) == (0, 0)


def test_ocr_result_distinguishes_empty_from_failed() -> None:
    """「图上没字」与「OCR 没跑成」必须是两种结果（否则会静默降级）。"""
    from src.sub_agents.vision_agent.ocr_runner import run_ocr

    ok_empty = OcrResult(text="", ok=True)
    broken = OcrResult(ok=False, reason="缺 chi_sim 字库")
    assert ok_empty.ok and not broken.ok and broken.text == ""
    real = run_ocr(_MIN_PNG)          # 真跑一次：纯色图上不会有字，但 OCR 本身是成功的
    assert isinstance(real, OcrResult)


# =============================================================================
# 4. 节点编排（monkeypatch 替身，不打真 API）
# =============================================================================


def _fake_model(image: VisionImage):
    """造一个假的视觉模型：``with_structured_output(...).invoke(...)`` 返回给定结论。"""
    class _Chain:
        def invoke(self, _messages):
            return image

    class _Model:
        def with_structured_output(self, _schema):
            return _Chain()

    return _Model()


@requires_key
def test_empty_image_refs_skips_everything(monkeypatch) -> None:
    """空 image_refs：不读图、不 OCR、不调模型；形状恒定。"""

    def boom(*_a, **_k):
        raise AssertionError("空图不应调用视觉模型")

    monkeypatch.setattr(vp, "understand_image", boom)
    out = vn.vision_node({"image_refs": []})
    assert out["image_refs"] == []
    assert out["visual_description"] == ""
    assert _as_findings(out["visual_findings"]).images == []


@requires_key
def test_useful_ocr_text_is_passed_to_model_but_never_skips_it(tmp_path, monkeypatch) -> None:
    """★ OCR 只做辅助：有用信息会进提示词，但**模型照样必须被调用**。"""
    ref = tmp_path / "screen.png"
    ref.write_bytes(_MIN_PNG)

    monkeypatch.setattr(vp, "run_ocr", lambda data: OcrResult(text="FLOW SP 12.5 m3/h\nALARM FAL-104"))
    seen = {}
    monkeypatch.setattr(vc, "vision_model", _fake_model(VisionImage(
        natural_description="模型看图后的描述",
        observations=[Observation(target="顶部告警条", finding="红色告警条", polarity="abnormal")],
        limitations=[],
    )))

    original = vp.understand_image

    def spy(data, content_type, ocr_text, state):
        seen["ocr_text"] = ocr_text
        return original(data, content_type, ocr_text, state)

    monkeypatch.setattr(vp, "understand_image", spy)
    out = vn.vision_node({"device_id": "PUMP-TEST", "image_refs": [str(ref)]})

    assert "ALARM FAL-104" in seen["ocr_text"], "有用文本被塞进了提示词"
    image = _as_findings(out["visual_findings"]).images[0]
    assert image.natural_description == "模型看图后的描述", "描述来自模型"
    assert image.observations[0].polarity == "abnormal"


@requires_key
def test_ocr_failure_is_reported_but_vision_still_runs(tmp_path, monkeypatch) -> None:
    """★ OCR 挂掉时：如实上报 + **照常调视觉模型**（不静默降级、也不放弃）。"""
    ref = tmp_path / "screen.png"
    ref.write_bytes(_MIN_PNG)

    monkeypatch.setattr(vp, "run_ocr", lambda data: OcrResult(ok=False, reason="缺 chi_sim 字库"))
    monkeypatch.setattr(vc, "vision_model", _fake_model(VisionImage(
        natural_description="纯视觉结论", observations=[], limitations=[])))

    image = vp.analyze_one(str(ref), {"device_id": "PUMP-TEST"})
    assert image.natural_description == "纯视觉结论"
    assert any("OCR 未生效" in lim for lim in image.limitations), "必须如实写进限制项"
    assert "OCR 未生效" in compose_description([str(ref)], [image]), "也要出现在文本里"


@requires_key
def test_compose_description_format() -> None:
    """每图一段 + `[图N] 图名` 锚点 + limitations 折成"本图未核验"。"""
    refs = ["/a/b/leak_close.png", "http://host/img/gauge_low.png"]
    images = [
        VisionImage(natural_description="压盖下方有深色液体沿底座流淌。",
                    observations=[Observation(target="密封压盖", finding="液体流淌", polarity="abnormal")],
                    limitations=["液体成分无法判断", "轴封形式被遮挡"]),
        VisionImage(natural_description="表盘量程 0～1.6 MPa，指针约 0.28 MPa。"),
    ]
    text = compose_description(refs, images)

    assert text.startswith("[图1] leak_close.png\n")
    assert "[图2] gauge_low.png" in text, "URL 也要只取图名做锚点"
    assert "本图未核验：液体成分无法判断；轴封形式被遮挡" in text
    assert text.count("本图未核验") == 1, "没有 limitations 的图不该出现这行"
    assert "\n\n" in text, "多图之间要空行分隔"


@requires_key
def test_batch_keeps_good_images_when_one_fails(tmp_path, monkeypatch) -> None:
    """★ 回归：一张坏图**不许**让整批一起失败（旧版会把已成功的结论全丢掉）。"""
    good = tmp_path / "good.png"
    good.write_bytes(_MIN_PNG)

    def fake_analyze_one(ref, state):
        if "坏图" in ref:
            raise FileNotFoundError(f"找不到图片文件: {ref}")
        return VisionImage(natural_description=f"{ref} 的结论")

    monkeypatch.setattr(vn, "analyze_one", fake_analyze_one)
    out = vn.vision_node({"image_refs": [str(good), "坏图.png", str(good)]})

    images = _as_findings(out["visual_findings"]).images
    assert len(images) == 3, "每张图都要有一条结论（失败的用占位）"
    assert images[1].natural_description.startswith(FAILED_PREFIX)
    assert images[0].natural_description.endswith("的结论"), "好图的结论必须还在"
    assert str(good) in out["visual_description"] and "坏图.png" in out["visual_description"]


@requires_key
def test_failed_image_reason_is_in_box_and_text(monkeypatch) -> None:
    """坏图要能溯源：原因写在 limitations + 文本里，且靠前缀能识别出失败。"""

    def boom(ref, state):
        raise ValueError("不支持的图片格式（需要 JPEG / PNG / GIF / WebP）")

    monkeypatch.setattr(vn, "analyze_one", boom)
    out = vn.vision_node({"image_refs": ["bad.png"]})

    failed = _as_findings(out["visual_findings"]).images[0]
    assert failed.natural_description.startswith(FAILED_PREFIX)
    assert failed.observations == []
    assert any("不支持的图片格式" in lim for lim in failed.limitations)
    text = out["visual_description"]
    assert "bad.png" in text and "识别失败" in text and "不支持的图片格式" in text


@requires_key
def test_all_images_failed_keeps_shape(monkeypatch) -> None:
    """全坏时形状照样恒定：结论条数 = 图片数，每条都是失败占位。"""
    monkeypatch.setattr(vn, "analyze_one", lambda ref, state: (_ for _ in ()).throw(OSError("坏")))
    out = vn.vision_node({"image_refs": ["a.png", "b.png"]})

    images = _as_findings(out["visual_findings"]).images
    assert len(images) == 2
    assert all(i.natural_description.startswith(FAILED_PREFIX) for i in images)
    assert out["visual_description"].count("识别失败") == 2


# =============================================================================
# 5. 重试：单层 + 可恢复错误重试 / 硬错误立刻抛
# =============================================================================


@requires_key
def test_vision_retry_is_single_layer() -> None:
    """★ 重试只有一层：`vision_model` 关掉了底层 SDK 自己的重试。

    不关的话是双层叠加（SDK 默认 2 × 我们的 3 轮 = 最坏 9 次真实请求，
    每次都是完整图文请求、会重复计费，而且日志里看不见真实次数）。
    """
    from src.utils.llm_client import vision_model

    assert vision_model.max_retries == 0, "langchain 层要显式关掉重试"
    assert vision_model.root_client.max_retries == 0, "底层 SDK 客户端也必须关掉"
    assert vc.VISION_MAX_ATTEMPTS >= 1, "我们这一层必须自己负责重试"


@requires_key
def test_vision_model_retries_transient_failure_then_succeeds(monkeypatch) -> None:
    """★ 网络波动要重试，而不是直接记失败（更不允许退化成仅 OCR）。"""
    class _FlakyChain:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("Connection reset by peer")
            return VisionImage(natural_description="重试后成功")

    chain = _FlakyChain()
    monkeypatch.setattr(vc, "vision_model", type("M", (), {
        "with_structured_output": lambda self, schema: chain})())
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # 别真等 1s+2s

    result = vc.understand_image(_MIN_PNG, "image/png", "", {"device_id": "P"})
    assert result.natural_description == "重试后成功"
    assert chain.calls == 3, "前两次失败后第三次成功"


@requires_key
def test_vision_model_does_not_retry_insufficient_balance(monkeypatch) -> None:
    """余额不足这类硬错误要**立刻抛**：重试只是白等，还掩盖真正原因。"""
    calls = {"n": 0}

    class _DeadChain:
        def invoke(self, _messages):
            calls["n"] += 1
            raise RuntimeError("Error code: 402 - Insufficient Balance")

    slept = {"n": 0}
    monkeypatch.setattr(vc, "vision_model", type("M", (), {
        "with_structured_output": lambda self, schema: _DeadChain()})())
    monkeypatch.setattr(time, "sleep", lambda _s: slept.__setitem__("n", slept["n"] + 1))

    with pytest.raises(RuntimeError, match="Insufficient Balance"):
        vc.understand_image(_MIN_PNG, "image/png", "", {"device_id": "P"})
    assert calls["n"] == 1, "硬错误只调一次"
    assert slept["n"] == 0, "不该退避等待"


# =============================================================================
# 6. 主图挂载 + 真实样例图
# =============================================================================


@requires_key
def test_step3_is_mounted_in_main_graph() -> None:
    from src.orchestrator.graph import build_main_graph

    app = build_main_graph()
    assert "step3" in app.get_graph().nodes


def test_sample_images_exist() -> None:
    """样例图在不在（不在只说明没放图，进不出 git 都无所谓）。"""
    if not _sample_images():
        pytest.skip(f"{_SAMPLE_ROOT} 下还没有样例图")


@requires_key
def test_parent_graph_exposes_contract_without_internal_fields() -> None:
    """★ 防回归：挂到父图后只该看到契约字段。

    这里不跑真主图（那会连带 Step 2 去查库），只挂一个"START → step3 → END"
    的迷你父图 —— 正好单独验证「节点产出 -> 父图状态」这一层过滤。
    """
    from langgraph.graph import END, START, StateGraph

    parent = StateGraph(DiagnosisState)
    parent.add_node("step3", vn.vision_node)
    parent.add_edge(START, "step3")
    parent.add_edge("step3", END)

    out = dict(parent.compile().invoke({"image_refs": []}))
    assert out["visual_description"] == ""
    assert _as_findings(out["visual_findings"]).images == []
    leaked = {"visual_rag_queries", "route", "image_kind", "ocr_text", "content_type",
              "current_ref", "need_vision", "route_used", "extracted_text"} & set(out)
    assert not leaked, f"这些键不该回流父图：{leaked}"


@requires_key
def test_sample_image_live_returns_description_and_box() -> None:
    """真实样例图跑通一张：描述带 [图N] 锚点，盒子里有对应的 3 字段结论。

    完整批量（9 张）请用 scripts/run_vision_samples.py，不要塞进 pytest —— 那是真花钱的。
    """
    images_path = _sample_images()
    if not images_path:
        pytest.skip(f"{_SAMPLE_ROOT} 下还没有样例图（放图后本用例自动生效）")

    first = images_path[0]
    out = vn.vision_node({"device_id": "PUMP-IS100-80-160-01",
                          "alarm_code": "FAL-104", "image_refs": [str(first)]})

    text = out["visual_description"]
    assert text.startswith(f"[图1] {first.name}"), f"锚点不对：{text[:60]!r}"

    findings = _as_findings(out["visual_findings"])
    assert len(findings.images) == 1, "一张图一份结论"
    image = findings.images[0]
    assert image.natural_description.strip(), "描述不能为空"
    # ★ 文本必须由盒子派生
    assert image.natural_description in text
    for obs in image.observations:
        assert obs.polarity in {"abnormal", "normal", "unknown"}
        assert obs.target.strip() and obs.finding.strip()
