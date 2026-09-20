#!/usr/bin/env python3
"""Step 3 视觉子图 · 样例图跑批脚本（走**真实链路**，会调 DeepSeek 视觉模型）。

用途：把 ``data/sample_inputs/``（含子目录）下的现场图**逐张**喂给真实子图
``src/sub_agents/vision_agent/vision_graph.py``，并打印每张图的：
    ① OCR 体检结论（这张图为什么走了/没走"仅 OCR 省钱"）
    ② 结构化结论（basis / 种类 / 置信度 / 观测 / 未核验项）—— 机器读的那一半
    ③ 描述文本（每图一段 + [图N] 锚点 + "本图未核验"）—— 给人读的那一半
    ④ 末尾汇总表（成功/失败、逐图 basis、种类、置信度、异常条数、字数）

与 ``pytest`` 的分工：
    · 本脚本 → 批量跑真实 API、给人看结果（慢、要花钱，**不进 CI**）
    · tests/test_vision_agent.py → 只挑 1 张图跑通契约（快）

★ 本脚本对每张图单独 ``try/except``，**这是脚本的体贴，不是生产代码的行为**：
  生产链路里一张坏图会掀翻整张 LangGraph（属于已知待办）。

用法（在项目根执行）：
    /home/yali_ai/work_file/.venv/bin/python scripts/run_vision_samples.py
    ... --dir data/sample_inputs/vision      # 只跑某个目录
    ... --device-id PUMP-IS100-80-160-01 --alarm-code FAL-104
    ... --limit 2 --json /tmp/vision_report.json --fail-fast

退出码：0 = 全部成功；1 = 有图失败（或 --fail-fast 提前中止）；2 = 前置条件不满足。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

#: 让脚本能以 `python scripts/xxx.py` 的方式直接跑（把项目根加进 import 路径）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: 认得的图片扩展名（真正判类型靠文件头，这里只用来筛文件名）
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
#: 默认扫描目录（递归）
DEFAULT_DIR = PROJECT_ROOT / "data" / "sample_inputs"


def _display(path: Path) -> str:
    """把路径显示得短一点：项目内的显示相对路径，项目外的显示绝对路径。

    参数：
        path: 任意文件路径。

    返回：
        项目根之下的路径 → 相对路径字符串（如 ``data/sample_inputs/vision/a.png``）；
        项目之外（如 ``--dir /tmp/xxx``）→ 原样的绝对路径。

    用途：
        纯粹为了日志好读，不参与任何业务判断。
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _ocr_status() -> str:
    """报告 OCR 后端是否可用（一句话），用来解释跑批结果里的 route/basis。

    参数：
        无。

    返回：
        一句中文状态描述，三种情况：
            "未装 pytesseract → run_ocr() 恒返回空串，所有图都会兜底走视觉模型"
            "可用（tesseract 4.1.1；字库 chi_sim, eng, osd）"
            "pytesseract 已装但 tesseract 不可用（<异常类型>: <原因>）"

    为什么要单独报这个：
        ``run_ocr`` 失败时静默返回空串。只看结果会以为"这些图本来就
        没有文字"，其实可能是 OCR 后端没装好 —— 这行输出能立刻分辨。
    """
    try:
        import pytesseract
    except ImportError:
        return "未装 pytesseract → run_ocr() 恒返回空串，所有图都会兜底走视觉模型"
    try:
        version = pytesseract.get_tesseract_version()
        langs = ", ".join(pytesseract.get_languages(config="")) or "无"
        return f"可用（tesseract {version}；字库 {langs}）"
    except Exception as exc:  # noqa: BLE001 二进制缺失/权限问题都算不可用
        return f"pytesseract 已装但 tesseract 不可用（{type(exc).__name__}: {exc}）"


def _collect_images(root: Path, limit: int | None) -> list[Path]:
    """递归收集目录下的图片文件（按路径排序，保证跑批顺序稳定）。

    参数：
        root:  要扫描的目录。
        limit: 只取前 N 张；传 None 表示不限。

    返回：
        图片路径列表（``Path`` 对象，已排序）。目录不存在或没有图片时返回空列表。
    """
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return files[:limit] if limit else files


def _print_ocr_verdict(ref: Path) -> None:
    """打印一张图的 **OCR 体检**结果（不含模型调用）。

    参数：
        ref: 图片路径。

    返回：
        无返回值，直接打印。形如：

            OCR 体检 : 4/6 可信→提示词 | 392 字 / 26 行 / 置信度 68 / 单位1 位号0 字段名5
                       未过项: 单位≥2 或 位号≥1、置信度≥70

    打这一行的目的：
        一眼看出这张图的 OCR 文本被判成了**噪声**还是**有用信息** ——
        是字数不够、还是没数字、还是没认到单位/位号。

    注意：
        这里会**再跑一次 Tesseract**（图小、1 秒级）只为诊断；
        生产链路里体检是嵌在 ``analyze_one`` 内部的，只跑一次。
        体检本身失败不影响主流程（只打一行"失败"）。
    """
    from src.sub_agents.vision_agent.image_tools import assess_ocr, read_image_bytes, resize_for_api

    try:
        verdict = assess_ocr(resize_for_api(read_image_bytes(str(ref))))
    except Exception as exc:  # noqa: BLE001 体检失败不该影响主流程
        print(f"  OCR 体检 : 失败（{type(exc).__name__}: {exc}）")
        return
    mark = "有用→当提示词" if verdict.credible else "噪声→丢弃"
    detail = (f"{verdict.chars} 字 / {verdict.digits} 数字 / "
              f"单位{len(verdict.units)} 位号{len(verdict.tags)} 字段名{len(verdict.keys)}")
    print(f"  OCR 体检 : {mark} | {detail}")


def _as_findings(value):
    """把子图返回的 ``visual_findings`` 统一成 ``VisionFindings`` 实例。

    参数：
        value: 从 ``invoke`` 结果里取出的 ``visual_findings``
               —— 可能是 ``VisionFindings`` 实例，也可能是等价的 dict
               （取决于 LangGraph 内部是否已做 pydantic 校验）。

    返回：
        ``VisionFindings`` 实例。``value`` 为 None 时返回空盒子。

    为什么需要：
        脚本不应该关心"LangGraph 这次给的是模型还是 dict"，
        统一入口能让后面的 ``.images[0]`` 写得干净、也不会因版本差异而崩。
    """
    from src.schemas.vision import VisionFindings

    if value is None:
        return VisionFindings()
    return value if isinstance(value, VisionFindings) else VisionFindings.model_validate(value)


def _print_image(image) -> None:
    """打印一张图的**结构化结论**（机器读的那一半）。

    参数：
        image: ``schemas.vision.VisionImage``。

    返回：
        无返回值，直接打印，形如：

            basis=vision  kind=leak_or_stain  conf=0.82  obs=7  abnormal=3
              obs[1] abnormal: 泵轴密封压盖 — 压盖与轴套交界处有深色油性液体附着…
              ! 未核验: 无 OCR 结果，画面中未见可读铭牌、表计读数或屏显信息
    """
    abnormal = sum(1 for o in image.observations if o.polarity == "abnormal")
    print(f"  basis={image.basis}  kind={image.image_kind}  conf={image.confidence}  "
          f"obs={len(image.observations)}  abnormal={abnormal}")
    for i, obs in enumerate(image.observations, 1):
        q = obs.quantity
        qs = f"  quantity={q.name}={q.value}{q.unit or ''}" if q else ""
        print(f"    obs[{i}] {obs.polarity}: {obs.target} — {obs.finding}{qs}")
    for lim in image.limitations:
        print(f"    ! 未核验: {lim}")


def _print_description(text: str) -> None:
    """打印子图交付的那段描述文本（给人读的那一半，也就是 Step4 拿去检索的东西）。

    参数：
        text: ``visual_description`` 的内容（多图时含多段）。

    返回：
        无返回值，直接打印；空文本打印"（空）"提示。
    """
    if not text.strip():
        print("  描述      : （空）")
        return
    for line in text.splitlines():
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    """脚本入口：解析参数 → 逐图跑真实子图 → 打印结果与汇总。

    参数：
        argv: 命令行参数列表；传 None 时取 ``sys.argv[1:]``（方便测试注入）。

    返回（退出码）：
        0 = 全部成功
        1 = 至少一张图失败（或 ``--fail-fast`` 提前中止）
        2 = 前置条件不满足（没配 DEEPSEEK_API_KEY / 目录不存在 / 目录里没图片）

    命令行参数：
        --dir PATH      图片目录（递归查找），默认 data/sample_inputs
        --device-id STR 传给提示词的设备号，默认 PUMP-IS100-80-160-01
        --alarm-code STR 传给提示词的告警码，默认 FAL-104
        --limit N       只跑前 N 张
        --json PATH     把完整结果（描述 + 结构化结论）写成 JSON
        --fail-fast     第一张失败就退出

    副作用：
        · 会调用真实大模型接口（要花钱、要联网）
        · 每张图会额外跑一次 Tesseract 用于打印体检结论
        · 打印大量内容到 stdout
    """
    parser = argparse.ArgumentParser(description="Step 3 视觉子图样例图跑批")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help=f"图片目录（递归查找），默认 {DEFAULT_DIR}")
    parser.add_argument("--device-id", default="PUMP-IS100-80-160-01")
    parser.add_argument("--alarm-code", default="FAL-104")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 张")
    parser.add_argument("--json", type=Path, default=None, help="把完整结果写成 JSON")
    parser.add_argument("--fail-fast", action="store_true", help="第一张失败就退出")
    args = parser.parse_args(argv)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("✗ 没有 DEEPSEEK_API_KEY。它写在 ~/.bashrc 里，请用**交互式登录终端**跑，")
        print("  或先执行： export DEEPSEEK_API_KEY=sk-xxxx")
        return 2

    if not args.dir.is_dir():
        print(f"✗ 目录不存在：{args.dir}")
        return 2

    images = _collect_images(args.dir, args.limit)
    if not images:
        print(f"✗ {args.dir} 下没有图片（支持 {', '.join(sorted(IMAGE_SUFFIXES))}）")
        return 2

    print("=" * 78)
    print(f"Step 3 真实链路跑批 | {len(images)} 张图 | 目录 {args.dir}")
    print(f"设备 {args.device_id} | 告警码 {args.alarm_code}")
    print(f"OCR 后端：{_ocr_status()}")
    print("=" * 78)

    # 延迟导入：没有 Key 时上面就已退出，不会在这里炸出难懂的 ValidationError
    from src.sub_agents.vision_agent.vision_graph import vision_agent_graph

    report: list[dict] = []
    failed = 0
    for idx, ref in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {_display(ref)}")
        _print_ocr_verdict(ref)
        started = time.time()
        try:
            out = dict(vision_agent_graph.invoke({
                "device_id": args.device_id,
                "alarm_code": args.alarm_code,
                "image_refs": [str(ref)],
            }))
        except Exception as exc:  # noqa: BLE001 脚本兜底，便于一次跑完看全貌
            failed += 1
            print(f"  ✗ {type(exc).__name__}: {exc}")
            report.append({"image": str(ref), "ok": False,
                           "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                break
            continue

        text = out.get("visual_description") or ""
        elapsed = time.time() - started
        findings = _as_findings(out.get("visual_findings"))
        if not text.strip() or not findings.images:
            failed += 1
            print(f"  ✗ {elapsed:.1f}s 返回为空（视作失败）")
            report.append({"image": str(ref), "ok": False, "error": "空描述/空盒子"})
            if args.fail_fast:
                break
            continue

        image = findings.images[0]
        if image.basis == "failed":
            # 逐图兜底的产物：链路没崩，但这张图确实没识别成功 —— 仍算失败
            failed += 1
            reason = image.limitations[0] if image.limitations else "未知原因"
            print(f"  ✗ {elapsed:.1f}s 该图处理失败：{reason}")
            report.append({"image": str(ref), "ok": False, "error": reason,
                           "result": image.model_dump()})
            if args.fail_fast:
                break
            continue

        print(f"  ✓ {elapsed:.1f}s")
        _print_image(image)
        print("  描述（给人看）:")
        _print_description(text)
        report.append({"image": str(ref), "ok": True, "seconds": round(elapsed, 1),
                       "visual_description": text,
                       "result": image.model_dump(),
                       "未核验行数": text.count("本图未核验")})

    # ---------------- 汇总 ----------------
    ok = [r for r in report if r["ok"]]
    print("\n" + "=" * 78)
    print(f"汇总：成功 {len(ok)} / 失败 {failed}")
    print(f"{'图片':44} {'basis':10} {'kind':18} {'conf':>5} {'abn':>4} {'字数':>6}")
    for row, ref in zip(report, images):
        if not row["ok"]:
            print(f"{ref.name:44} {'ERROR':>10} {row['error'][:40]}")
            continue
        image = row["result"]
        abnormal = sum(1 for o in image["observations"] if o["polarity"] == "abnormal")
        print(f"{ref.name:44} {image['basis']:10} {image['image_kind']:18} "
              f"{image['confidence']:>5} {abnormal:>4} {len(row['visual_description']):>6}")
    print("=" * 78)

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON 已写入：{args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
