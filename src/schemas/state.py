"""
CentriPump-PHM 全局工作流状态定义。

设计目标
--------
本文件定义整个故障诊断流程共享的 DiagnosisState。

状态按照业务职责划分为 9 个一级区域：

    DiagnosisState
    ├── context      # 一次诊断任务的基础上下文 / 输入
    ├── data         # Step 2：时序数据分析
    ├── vision       # Step 3：现场图片分析
    ├── manual       # Step 4：手册 / RAG
    ├── reasoning    # Step 5：故障归因
    ├── safety       # Step 6：确定性安全门禁
    ├── human        # 人工介入
    ├── delivery     # Step 7：工单与证据溯源
    └── workflow     # 流程级元数据

这样做的核心目的不是“增加抽象”，而是让 State 本身就能体现系统架构。

例如：

    state.data.metrics
    state.vision.findings
    state.manual.evidences
    state.reasoning.hypotheses
    state.safety.decision

比：

    state.calculated_metrics
    state.vision_findings
    state.manual_evidence
    state.hypotheses
    state.guard_decision

更容易阅读，也更容易明确每个 Agent 的职责边界。


重要约束
--------
1. State 中的数据最终必须能够被 LangGraph Checkpoint 使用的
   msgpack 序列化。

   因此只允许：

       str
       int
       float
       bool
       None
       list
       dict

   不允许：

       numpy.float64
       numpy.ndarray
       pandas.DataFrame
       datetime
       ORM 对象
       SQLAlchemy Session
       数据库 Engine
       LLM Client
       文件对象
       等等。


2. 大型数据不直接进入 State。

   例如：

       SCADA 原始时序数据
       原始图片二进制
       完整手册正文
       完整工单正文

   都不应该进入 checkpoint。

   State 中只保存引用：

       telemetry_ref
       image_refs
       manual evidence 的 doc_id/chunk_id
       report_uri
       report_ref


3. trace_id、时间戳等“一次生成”的值不能使用随机 default_factory。

   例如不能：

       trace_id: str = Field(default_factory=lambda: uuid4().hex)

   因为 State 在 LangGraph 生命周期中可能被多次物化。

   trace_id 应该由入口显式创建并写入 State。


4. SafetyState 只能由确定性规则代码产生。

   LLM 可以提供：

       reasoning

   但不能直接决定：

       safety.decision
       safety.risk_level

   这些必须来自 rules/ 中的确定性逻辑。


5. 每个 Agent 尽量只负责自己对应的一级 State。

   例如：

       Data Agent
           -> data

       Vision Agent
           -> vision

       Manual Agent
           -> manual

       Reasoner
           -> reasoning

       Safety Guard
           -> safety

       Reporter
           -> delivery
"""


from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# 一、公共基础设施
# =============================================================================

# LangGraph Checkpoint 最终需要处理的基础类型。
#
# 注意：
#   bool 必须单独包含。
#   Python 中 bool 是 int 的子类，但这里我们使用 type(value) 精确判断，
#   所以两者都列出来。
_ALLOWED_VALUE_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
    list,
    dict,
)


def _assert_checkpoint_safe(value: Any, path: str) -> None:
    """
    递归检查一个值是否由 checkpoint 可以安全处理的原生类型组成。

    为什么需要这个函数？
    --------------------
    如果把 numpy.float64 / datetime / ORM 对象等塞进 State，
    很可能不是在业务代码执行的时候报错，而是在 checkpoint 保存时才报错。

    例如：

        state.data.metrics["rms"] = numpy.float64(1.25)

    业务代码可能完全正常。

    直到 LangGraph checkpoint：

        TypeError:
            Type is not msgpack serializable: numpy.float64

    这种错误很难排查。

    所以这里在 Pydantic Model 完成校验时主动检查。

    为什么使用 type(value) 而不是 isinstance？
    ------------------------------------------
    因为某些第三方数值类型可能是 Python 基础类型的子类。

    我们希望的是：

        真的就是 float
        真的就是 int

    而不是：

        “看起来像 float”

    tuple 也主动拒绝。

    虽然某些情况下 msgpack 可以处理 tuple，
    但经过序列化 / 反序列化之后可能变成 list，
    会导致 State 往返后的类型不稳定。

    所以项目统一约定：

        tuple -> list
    """

    # 【新增】BaseModel（嵌套 State 子模型）：
    # 先转成原生 dict，再递归检查。
    #
    # 为什么需要这个分支？
    # --------------------
    # State 中的嵌套子模型（例如 DataQuality / TelemetryRef）
    # 在 Pydantic 校验完成后仍然是 BaseModel 实例，
    # 而不是 dict。
    #
    # 如果不处理 BaseModel，
    # 所有包含嵌套子模型的 State
    # 都会在构造时被误判为“无法安全进入 checkpoint”。
    #
    # model_dump() 保留 python 原生类型，
    # 不会把数值转成 JSON 字符串，
    # 因此 numpy.float64 等非法类型仍然能被检查出来。
    if isinstance(value, BaseModel):
        _assert_checkpoint_safe(value.model_dump(), path)
        return

    # dict：递归检查所有 key 和 value。
    if type(value) is dict:
        for key, item in value.items():
            # 【新增】msgpack 的 map key 必须是 str，
            # 所以 key 的类型也需要检查。
            if type(key) is not str:
                raise TypeError(
                    f"状态字段 {path} 的 dict key 类型 "
                    f"{type(key).__name__} 无法安全进入 checkpoint。"
                    "dict key 必须转换为 str。"
                )
            _assert_checkpoint_safe(item, f"{path}.{key}")
        return

    # list：递归检查每一个元素。
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_checkpoint_safe(item, f"{path}[{index}]")
        return

    # 基础类型直接通过。
    if type(value) in _ALLOWED_VALUE_TYPES:
        return

    raise TypeError(
        f"状态字段 {path} 的值类型 "
        f"{type(value).__name__} 无法安全进入 checkpoint。"
        "请先转换成原生类型："
        "str / int / float / bool / None / list / dict。"
        "numpy 数值请先转换，datetime 请转换为 ISO 字符串，"
        "tuple 请转换为 list，大型数据请只保存引用。"
    )


class StateModel(BaseModel):
    """
    所有 State 子模型的公共基类。

    这里统一放两个规则：

    1. extra="forbid"

       禁止 Agent 往 State 中写入没有声明的字段。

       例如：

           data.calculated_metric

       写成：

           data.calculated_metrics

       那么应该立即报错，而不是静默接受。

    2. checkpoint 类型安全检查

       每个 State 子模型创建完成之后，递归检查里面的数据。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="after")
    def _checkpoint_safety(self):
        """
        检查当前 State Model 内部所有字段的值。

        注意：
        这里只检查“值”，不会限制字段声明必须是什么类型。
        例如：

            dict[str, Any]

        仍然可以使用。

        真正进入 checkpoint 的最终值必须经过这里的检查。
        """

        for field_name in type(self).model_fields:
            value = getattr(self, field_name)

            _assert_checkpoint_safe(
                value,
                field_name,
            )

        return self


# =============================================================================
# 二、Context —— Step 1 / 整个诊断任务的基础上下文
# =============================================================================


class ImageRef(StateModel):
    """
    现场图片引用。

    图片本体不进入 LangGraph State。

    State 只保存：
        image_id
        uri
        source
        captured_at

    真正的图片可以存在：
        本地文件
        对象存储
        数据库
        归档系统

    image_id 是长期身份。
    uri 只是当前运行环境中的访问地址。
    """

    image_id: str = ""
    """图片唯一 ID。"""

    uri: str = ""
    """
    图片访问地址。

    可以是：
        本地路径
        对象存储 URL
        文件系统 URI
        归档系统定位符
    """

    source: str = ""
    """
    图片来源。

    例如：
        INSPECTION
        OPERATOR
        SCADA_EXPORT
    """

    captured_at: str = ""
    """
    图片拍摄时间。

    统一使用 ISO 字符串。
    未知时为空字符串。
    """


class ContextState(StateModel):
    """
    一次诊断任务的基础上下文。

    这里放的是：

        “这一次到底在诊断什么？”

    而不是某个 Agent 的分析结果。
    """

    # -------------------------------------------------------------------------
    # 诊断任务身份
    # -------------------------------------------------------------------------

    trace_id: str = ""
    """
    整个诊断流程的唯一 ID。

    它同时用于关联：

        LangGraph
        Log
        数据库
        归档数据
        工单

    注意：
    不使用 default_factory 自动生成。
    应由入口显式生成。
    """

    device_id: str = ""
    """正在诊断的设备编号。"""

    start_time: str = ""
    """诊断数据窗口起点，ISO 字符串。"""

    end_time: str = ""
    """诊断数据窗口终点，ISO 字符串。"""

    # -------------------------------------------------------------------------
    # 用户输入
    # -------------------------------------------------------------------------

    alarm_code: str = ""
    """
    用户提供的报警码。

    可以为空。
    """

    user_query: str = ""
    """
    用户原始问题。

    例如：

        “3号泵最近振动明显升高，帮忙判断原因。”

    原始问题应该保留，不要只保存 LLM 改写后的内容。
    """

    # -------------------------------------------------------------------------
    # 现场图片
    # -------------------------------------------------------------------------

    image_refs: list[ImageRef] = Field(
        default_factory=list,
    )
    """
    本次诊断使用的现场图片。

    图片是诊断输入，所以属于 context。

    注意：
        image_refs 只保存图片引用，
        不保存图片二进制。
    """

    # -------------------------------------------------------------------------
    # Step 1 Router 输出
    # -------------------------------------------------------------------------

    # 新增修改，删除路由原因route_reason，不再进行分流，而是交由每个专家自行判断，给出数据情况描述


# =============================================================================
# 三、Data —— Step 2 时序数据分析
# =============================================================================

class TelemetryRef(StateModel):
    """
    原始遥测数据的引用。

    原始 SCADA 数据不进入 checkpoint。

    凭这些信息，Data Agent 可以重新读取同一个数据窗口。
    """

    source: str = ""
    """
    数据来源。

    例如：

        scada_db.scada_telemetry
    """

    trace_id: str = ""
    """对应的诊断 trace_id。"""

    device_id: str = ""
    """设备编号。"""

    start_time: str = ""
    """数据窗口开始时间。"""

    end_time: str = ""
    """数据窗口结束时间。"""


class DataQuality(StateModel):
    """
    时序数据质量。

    用于区分：

        正常取得数据
        没有数据
        数据不完整
        尚未执行

    这对后续 Safety Guard 很重要。
    """

    status: Literal[
        "PENDING",
        "OK",
        "EMPTY",
        "PARTIAL",
    ] = "PENDING"

    total_points: int = 0
    """本次实际取得的数据点数量。"""

    reason: str = ""
    """
    数据质量异常原因。

    例如：

        “指定时间窗口内没有 SCADA 数据”
        “窗口前 20 分钟数据缺失”
    """


class AlarmState(StateModel):
    """
    报警码相关信息。

    原来这几个字段是：

        effective_alarm_codes
        last_alarm_codes
        all_alarm_codes_in_window

    它们本质上属于同一类信息，所以收拢成 alarms。
    """

    effective: str = "NONE"
    """
    当前分析认为有效的报警码组合。
    """

    last: str = "NONE"
    """
    时间窗口内最后出现的非 NONE 报警码组合。
    """

    all: list[str] = Field(
        default_factory=list,
    )
    """
    整个窗口内出现过的报警码。
    """

class DataDescription(StateModel):
    """
    Data Agent 对原始数据分析结果进行语义化后的单条描述。

    注意：
    这里描述的是“数据表现出来的现象”，
    而不是“故障原因”。

    例如：
        正确：
            “振动 RMS 在报警前 30 秒持续升高，并超过预警阈值。”

        不应该：
            “初步判断为轴承故障。”

    后者属于 Reasoning Agent 的因果分析职责。
    """

    # 【修改】这里继承 StateModel（原来是 BaseModel），
    # 与项目中其它 State 子模型保持一致，
    # 从而获得：
    #   1. extra="forbid"
    #   2. checkpoint 类型安全检查
    #
    # 不再单独写 model_config。

    type: Literal[
        "TREND",       # 趋势特征
        "ANOMALY",     # 异常特征
        "THRESHOLD",   # 阈值越界
        "CORRELATION", # 多指标之间的相关变化
        "ALARM",       # 报警相关特征
        "OTHER",       # 其他数据特征
    ] = Field(
        description="该语义特征的类型。"
    )

    description: str = Field(
        min_length=1,
        description=(
            "对数据现象的自然语言描述。"
            "只能描述观测到的数据特征，不得直接进行故障归因。"
        ),
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "支撑该描述的原始数据字段或计算指标名称。"
            "例如：['vibration_rms', 'temperature']。"
        ),
    )


class DataState(StateModel):
    """
    Step 2 Data Agent 的全部输出。

    Data Agent 的职责：

        原始 SCADA 数据
            ↓
        数据清洗
            ↓
        统计 / FFT / 突变检测
            ↓
        降维后的事实
            ↓
        DataState

    注意：
    Data Agent 不负责最终故障归因。
    """

    telemetry_ref: TelemetryRef = Field(
        default_factory=TelemetryRef,
    )
    """
    原始遥测数据引用。

    不保存原始数据本体。
    """

    quality: DataQuality = Field(
        default_factory=DataQuality,
    )
    """数据质量。"""

    metrics: dict[str, Any] = Field(
        default_factory=dict,
    )
    """
    计算后的统计指标。

    例如：

        {
            "overall": {
                "rms": 12.3,
                "mean": 10.2,
            },
            "phases": [...]
        }

    这里允许 Any，但实际写入的数据必须经过 checkpoint
    原生类型检查。

    不允许：

        numpy.float64
        numpy.ndarray
        pandas.Series
    """

    threshold_flags: list[str] = Field(
        default_factory=list,
    )
    """
    数据分析阶段发现的阈值异常。

    例如：

        [
            "VIBRATION_RMS_HIGH",
            "BEARING_TEMP_HIGH",
        ]

    这些是“事实 / 判据”，不是最终根因。
    """

    alarms: AlarmState = Field(
        default_factory=AlarmState,
    )
    """报警码相关信息。"""

    descriptions: list[DataDescription] = Field(
        default_factory=list,
        description=(
            "Data Agent LLM 根据 metrics、threshold_flags 和 alarms "
            "提取出的结构化数据语义特征。"
            "每一条描述都必须能够通过 evidence 回溯到具体数据。"
        ),
    )



# =============================================================================
# 四、Vision —— Step 3 视觉分析
# =============================================================================


class VisionFinding(StateModel):
    """
    单条视觉缺陷发现。

    每条 Finding 必须能够回溯到具体 image_id。
    """

    image_id: str = ""
    """产生该结论的图片 ID。"""

    defect_type: str = ""
    """
    缺陷类型。

    例如：

        CRACK
        WEAR
        LEAK
        SCALE
    """

    severity: Literal[
        "MINOR",
        "MODERATE",
        "SEVERE",
    ] = "MINOR"
    """缺陷严重程度。"""

    location: str = ""
    """缺陷所在设备部件 / 位置。"""

    confidence: float = 0.0
    """
    视觉模型自身对该发现的置信度。

    注意：
    这是视觉分析结果的一部分。

    它不能直接等价于整个故障诊断的最终置信度。
    """

    evidence: str = ""
    """
    视觉判断依据。

    例如：

        “叶轮边缘存在明显不规则缺口及局部磨损痕迹。”
    """


class VisionState(StateModel):
    """
    Step 3 Vision Agent 输出。
    """

    status: Literal[
        "PENDING",
        "NO_IMAGE",
        "NO_DEFECT",
        "DEFECT_FOUND",
        "FAILED",
    ] = "PENDING"
    """
    视觉分析状态。

    必须区分：

        NO_IMAGE
            没有图片。

        NO_DEFECT
            有图片，并且分析后没有发现明显缺陷。

        FAILED
            有图片，但是视觉分析执行失败。

    这三种情况对后续推理的含义不同。
    """

    findings: list[VisionFinding] = Field(
        default_factory=list,
    )
    """逐条视觉缺陷发现。"""

    summary: str = ""
    """
    面向 Reasoner 的视觉语义摘要。

    findings：
        偏结构化、偏工单。

    summary：
        偏语义理解、偏推理。
    """


# =============================================================================
# 五、Manual —— Step 4 手册 / RAG
# =============================================================================


class ManualEvidence(StateModel):
    """
    一条手册证据。

    不把完整手册正文放入 State。

    只保存：

        文档身份
        chunk 身份
        章节
        页码
        短摘录
        检索得分
        检索通道

    真正的完整正文需要时再通过 doc_id / chunk_id 查询。
    """

    doc_id: str = ""
    """文档 ID。"""

    chunk_id: str = ""
    """知识库 chunk ID。"""

    section: str = ""
    """章节 / 条款号。"""

    page: int = 0
    """页码。"""

    quote: str = ""
    """
    短摘录。

    这里只保留能够帮助审计 / 工单阅读的短文本。
    不应该把整页手册复制进 checkpoint。
    """

    score: float = 0.0
    """检索得分。"""

    channel: Literal[
        "DENSE",
        "BM25",
        "HYBRID",
    ] = "HYBRID"
    """该证据来自哪个检索通道。"""


class ProcedureRequirement(StateModel):
    """
    手册中的规程要求。
    """

    requirement: str = ""
    """
    规程要求内容。

    例如：

        “检查轴承润滑状态后方可继续运行。”
    """

    source_ref: str = ""
    """
    来源引用。

    例如：

        manual_001#chunk_023
    """


class RecommendedAction(StateModel):
    """
    基于证据形成的建议动作。

    注意：
    “建议动作”不等于“安全放行”。

    最终是否允许执行相关操作，
    仍然需要经过 Safety Guard。
    """

    action: str = ""
    """建议动作。"""

    priority: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
    ] = "MEDIUM"
    """建议优先级。"""

    source_ref: str = ""
    """
    动作依据。

    可以指向：

        rule_id
        doc_id#chunk_id
        image_id
    """


class ManualState(StateModel):
    """
    Step 4 Manual Agent 输出。
    """

    status: Literal[
        "PENDING",
        "HIT",
        "NO_HIT",
        "FAILED",
    ] = "PENDING"
    """
    RAG 执行状态。

    NO_HIT：
        搜索成功，但没有找到相关内容。

    FAILED：
        RAG 系统本身执行失败。

    两者必须区分。
    """

    evidences: list[ManualEvidence] = Field(
        default_factory=list,
    )
    """真正被 Reasoner / Reporter 使用的手册证据。"""

    mechanism: str = ""
    """
    手册知识提炼后的故障机理。

    必须能够回指 evidences。
    """

    requirements: list[ProcedureRequirement] = Field(
        default_factory=list,
    )
    """规程要求。"""

    actions: list[RecommendedAction] = Field(
        default_factory=list,
    )
    """基于手册证据产生的建议动作。"""


# =============================================================================
# 六、Reasoning —— Step 5 故障归因
# =============================================================================


class Hypothesis(StateModel):
    """
    一条故障原因假设。

    即使某个假设最后被否定，也建议保留。

    因为最终工单可能需要展示：

        “排查过什么？”
        “为什么排除？”
    """

    cause: str = ""
    """故障原因。"""

    confidence_level: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
    ] = "LOW"
    """
    该假设本身的证据强度。

    注意：
    它不是 Safety Guard 的最终放行依据。
    """

    support_refs: list[str] = Field(
        default_factory=list,
    )
    """
    支持该假设的证据引用。

    例如：

        threshold:VIBRATION_RMS_HIGH
        image:IMG_001
        manual:MANUAL_001#CHUNK_03
    """

    contradicting_evidence: str = ""
    """
    与该假设矛盾的证据。

    如果没有，留空。
    """

    status: Literal[
        "SUPPORTED",
        "INSUFFICIENT",
        "CONFLICTED",
    ] = "INSUFFICIENT"
    """
    当前假设状态。
    """


class ReasoningState(StateModel):
    """
    Step 5 Reasoner 输出。

    Reasoner 可以调用 LLM。

    但它的输出最终还要经过 Step 6
    的确定性 Safety Guard。
    """

    root_cause: str = ""
    """
    当前认为最主要的根因。

    该字段应该由 Reasoner 产生。

    Reporter 可以使用，
    但 Safety Guard 不应该盲信。
    """

    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
    )
    """
    全部候选故障原因。

    包括：
        支持的
        证据不足的
        被冲突证据否定的
    """

    confidence: Literal[
        "PENDING",
        "LOW",
        "MEDIUM",
        "HIGH",
    ] = "PENDING"
    """
    整体归因置信度。

    使用分档而不是 float，
    方便确定性规则处理。
    """

    support_sources: list[str] = Field(
        default_factory=list,
    )
    """
    支撑最终判断的独立证据来源。

    例如：

        [
            "DATA",
            "VISION",
            "MANUAL",
        ]
    """

    summary: str = ""
    """
    整体归因链说明。

    用于 Reporter 生成工单正文。
    """

    insufficient_evidence: bool = False
    """
    是否明确判断当前证据不足以完成可靠归因。
    """

    conflicts: list[str] = Field(
        default_factory=list,
    )
    """
    当前发现的证据冲突。

    例如：

        “SCADA 振动数据支持轴承异常，
         但现场图片未发现对应磨损。”
    """

    used_inputs: list[str] = Field(
        default_factory=list,
    )
    """
    本次推理实际使用了哪些输入。

    例如：

        [
            "data",
            "vision",
            "manual",
        ]

    这个字段的意义是：
    防止 Reasoner 声称“综合了所有信息”，
    实际上某个 Agent 根本没有提供有效结果。
    """


# =============================================================================
# 七、Safety —— Step 6 确定性安全门禁
# =============================================================================


class SafetyState(StateModel):
    """
    Step 6 Safety Guard 输出。

    ★ 这是整个系统最重要的安全边界之一。

    这里的数据只能由：

        rules/safety_guard.py

    这样的确定性代码产生。

    不允许：

        LLM 直接生成 decision
        LLM 直接生成 risk_level

    Reasoner 可以说：

        root_cause = “轴承润滑不足”

    但最终：

        PASS
        MANUAL_REVIEW
        BLOCK
        INSUFFICIENT_DATA

    必须由确定性规则计算。
    """

    risk_level: str = ""
    """
    风险等级。

    具体值由：

        rules/ram_matrix.json

    定义。

    因为风险等级是配置驱动的，
    这里不强行使用 Literal。
    """

    decision: Literal[
        "PENDING",
        "PASS",
        "MANUAL_REVIEW",
        "BLOCK",
        "INSUFFICIENT_DATA",
    ] = "PENDING"
    """
    Safety Guard 最终决策。

    Reporter 应该只把 PASS 当作：
        “允许生成正式自动工单”

    其他状态应该进入对应的降级 / 人工流程。
    """

    rule_hits: list[str] = Field(
        default_factory=list,
    )
    """
    命中的机器可读规则。

    例如：

        [
            "HIGH_RISK_DEVICE",
            "LOW_CONFIDENCE",
            "MISSING_TELEMETRY",
        ]
    """

    reasons: list[str] = Field(
        default_factory=list,
    )
    """
    面向人类的规则说明。

    Reporter 可以直接用于工单。
    """


# =============================================================================
# 八、Human —— 人工介入
# =============================================================================


class HumanState(StateModel):
    """
    人工介入信息。

    Human 和 Safety 必须分开。

    Safety：

        “按照机器规则，现在是否允许继续？”

    Human：

        “人工审核之后，最终做了什么决定？”

    这是两个不同的概念。
    """

    decision: str = ""
    """
    人工最终决定。

    例如：

        允许继续
        停机检查
        转现场工程师
    """

    decided_at: str = ""
    """人工决定时间，ISO 字符串。"""

    reviewer: str = ""
    """
    审核人员。

    可以是：
        工号
        用户名
        姓名

    实际采用什么身份体系由业务系统决定。
    """

    comment: str = ""
    """人工审核备注。"""


# =============================================================================
# 九、Delivery —— Step 7 最终交付
# =============================================================================


class EvidenceRef(StateModel):
    """
    最终证据链中的一环。

    用来表达：

        最终结论
            ↓
        哪个步骤产生
            ↓
        哪个 State 字段
            ↓
        哪个原始证据
            ↓
        原始证据具体在哪里
    """

    stage: str = ""
    """
    产生该证据的步骤。

    例如：

        step2
        step3
        step4
        step5
        step6
    """

    field: str = ""
    """
    对应 State 字段。

    例如：

        data.threshold_flags
        vision.findings
        manual.evidences
        reasoning.hypotheses
        safety.rule_hits
    """

    source_id: str = ""
    """
    原始来源 ID。

    例如：

        rule_id
        doc_id#chunk_id
        image_id
    """

    locator: str = ""
    """
    进一步定位信息。

    例如：

        页码
        时间窗口
        图片区域
        SCADA 时间戳
    """


class DeliveryState(StateModel):
    """
    Step 7 Reporter 输出。

    注意：

    完整工单正文不进入 LangGraph State。

    State 只保存：

        工单身份
        工单位置
        工单 hash
        证据链

    完整工单可以放：

        数据库
        对象存储
        文件系统
        文档服务
    """

    diagnosed_at: str = ""
    """
    正式诊断完成时间。

    必须由 Step 7 显式写入。

    不使用 default_factory。
    """

    report_ref: str = ""
    """
    工单内容 hash。

    用于验证：

        当前工单内容
        是否就是当时生成的那一份。
    """

    report_uri: str = ""
    """
    工单实际存储位置。

    例如：

        数据库记录 ID
        对象存储 URI
        文件路径
    """

    traceability: list[EvidenceRef] = Field(
        default_factory=list,
    )
    """
    最终证据链。

    用于：

        审计
        人工复评
        故障复盘
        后续模型训练
    """


# =============================================================================
# 十、Workflow —— 流程级元数据
# =============================================================================


class WorkflowState(StateModel):
    """
    与具体业务 Agent 无关的流程级信息。

    这里故意保持很小。

    不要把：

        current_node
        last_node
        retry_count
        route_steps
        execution_history

    等大量 LangGraph 执行信息全部复制进业务 State。

    LangGraph 自身的 checkpoint / execution history
    已经承担了大量这类职责。
    """

    schema_version: str = "1.0"
    """
    State 契约版本。

    如果修改 State 的结构，
    应该同步升级版本。

    例如：

        1.0
        1.1
        2.0
    """

    degraded_steps: list[str] = Field(
        default_factory=list,
    )
    """
    已经执行，但发生降级的步骤。

    例如：

        [
            "step3",
            "step4",
        ]

    Step 5 / Step 6 / Step 7
    可以根据这个字段判断当前流程是否存在降级。
    """


# =============================================================================
# 十一、DiagnosisState —— LangGraph 全局状态
# =============================================================================


class DiagnosisState(StateModel):
    """
    CentriPump-PHM 的 LangGraph 全局状态。

    这是整个项目最重要的数据契约。

    整体结构：

        DiagnosisState
        │
        ├── context
        │     ├── trace_id
        │     ├── device_id
        │     ├── time window
        │     ├── user_query
        │     └── image_refs
        │
        ├── data
        │     ├── telemetry
        │     ├── quality
        │     ├── metrics
        │     └── alarms
        │
        ├── vision
        │     ├── status
        │     └── findings
        │
        ├── manual
        │     ├── evidences
        │     ├── mechanism
        │     └── requirements
        │
        ├── reasoning
        │     ├── root_cause
        │     ├── hypotheses
        │     └── confidence
        │
        ├── safety
        │     ├── risk_level
        │     ├── decision
        │     └── rule_hits
        │
        ├── human
        │     ├── decision
        │     └── reviewer
        │
        ├── delivery
        │     ├── report_ref
        │     ├── report_uri
        │     └── traceability
        │
        └── workflow
              ├── schema_version
              └── degraded_steps


    Agent 与 State 的关系：

        Step 1 Router
            ↓
        context

        Step 2 Data Agent
            ↓
        data

        Step 3 Vision Agent
            ↓
        vision

        Step 4 Manual Agent
            ↓
        manual

        Step 5 Reasoner
            ↓
        reasoning

        Step 6 Safety Guard
            ↓
        safety

        Human Review
            ↓
        human

        Step 7 Reporter
            ↓
        delivery

        workflow
            ↓
        所有节点共享
    """

    # -------------------------------------------------------------------------
    # 1. 基础上下文
    # -------------------------------------------------------------------------

    context: ContextState = Field(
        default_factory=ContextState,
    )

    # -------------------------------------------------------------------------
    # 2. Step 2 数据分析
    # -------------------------------------------------------------------------

    data: DataState = Field(
        default_factory=DataState,
    )

    # -------------------------------------------------------------------------
    # 3. Step 3 视觉分析
    # -------------------------------------------------------------------------

    vision: VisionState = Field(
        default_factory=VisionState,
    )

    # -------------------------------------------------------------------------
    # 4. Step 4 手册 / RAG
    # -------------------------------------------------------------------------

    manual: ManualState = Field(
        default_factory=ManualState,
    )

    # -------------------------------------------------------------------------
    # 5. Step 5 故障归因
    # -------------------------------------------------------------------------

    reasoning: ReasoningState = Field(
        default_factory=ReasoningState,
    )

    # -------------------------------------------------------------------------
    # 6. Step 6 确定性安全门禁
    # -------------------------------------------------------------------------

    safety: SafetyState = Field(
        default_factory=SafetyState,
    )

    # -------------------------------------------------------------------------
    # 7. 人工介入
    # -------------------------------------------------------------------------

    human: HumanState = Field(
        default_factory=HumanState,
    )

    # -------------------------------------------------------------------------
    # 8. Step 7 最终交付
    # -------------------------------------------------------------------------

    delivery: DeliveryState = Field(
        default_factory=DeliveryState,
    )

    # -------------------------------------------------------------------------
    # 9. 流程元数据
    # -------------------------------------------------------------------------

    workflow: WorkflowState = Field(
        default_factory=WorkflowState,
    )

    @model_validator(mode="after")
    def _checkpoint_safety(self) -> "DiagnosisState":
        """
        最终再对整个 DiagnosisState 做一次 checkpoint 安全检查。

        子模型本身已经检查过一次。

        这里再次检查的意义是：

        1. 防止运行过程中有人直接修改嵌套数据；
        2. 作为整个 DiagnosisState 的最终安全边界；
        3. 将来如果增加特殊字段，不容易漏掉检查。
        """

        for field_name in type(self).model_fields:
            value = getattr(self, field_name)

            _assert_checkpoint_safe(
                value,
                field_name,
            )

        return self


# =============================================================================
# 十二、初始 State 构造函数
# =============================================================================


def create_initial_state(
    *,
    trace_id: str,
    device_id: str,
    start_time: str,
    end_time: str,
    user_query: str = "",
    alarm_code: str = "",
    image_refs: list[ImageRef] | None = None,
) -> DiagnosisState:
    """
    创建一次新的诊断流程初始 State。

    为什么单独提供这个函数？
    ------------------------
    不把“默认值”和“初始化值”混在一起。

    例如：

        trace_id

    是一次诊断任务创建时产生的值。

    它不是 State 的“默认值”。

    正确流程应该是：

        API / main
            ↓
        uuid4()
            ↓
        create_initial_state(trace_id=...)
            ↓
        LangGraph

    而不是：

        DiagnosisState()
            ↓
        default_factory=uuid4()

    后者在某些 State 重建场景下容易产生
    “同一个逻辑流程多个 trace_id” 的问题。

    参数
    ----
    trace_id:
        外部生成的一次诊断唯一 ID。

    device_id:
        被诊断设备。

    start_time / end_time:
        SCADA 数据窗口。

    user_query:
        用户原始问题。

    alarm_code:
        用户提供的报警码。

    image_refs:
        现场图片引用。
    """

    # -------------------------------------------------------------------------
    # 处理图片列表
    #
    # 不直接保存调用方传进来的 None。
    # State 中统一使用 list。
    # -------------------------------------------------------------------------

    if image_refs is None:
        image_refs = []

    # -------------------------------------------------------------------------
    # 构造 State
    # -------------------------------------------------------------------------

    return DiagnosisState(
        context=ContextState(
            trace_id=trace_id,
            device_id=device_id,
            start_time=start_time,
            end_time=end_time,
            alarm_code=alarm_code,
            user_query=user_query,
            image_refs=image_refs,
        ),

        data=DataState(
            telemetry_ref=TelemetryRef(
                source="",
                trace_id=trace_id,
                device_id=device_id,
                start_time=start_time,
                end_time=end_time,
            ),
            quality=DataQuality(
                status="PENDING",
                total_points=0,
                reason="",
            ),
            metrics={},
            threshold_flags=[],
            alarms=AlarmState(
                effective="NONE",
                last="NONE",
                all=[],
            ),
            descriptions=[],
        ),

        vision=VisionState(
            status="PENDING",
            findings=[],
            summary="",
        ),

        manual=ManualState(
            status="PENDING",
            evidences=[],
            mechanism="",
            requirements=[],
            actions=[],
        ),

        reasoning=ReasoningState(
            root_cause="",
            hypotheses=[],
            confidence="PENDING",
            support_sources=[],
            summary="",
            insufficient_evidence=False,
            conflicts=[],
            used_inputs=[],
        ),

        safety=SafetyState(
            risk_level="",
            decision="PENDING",
            rule_hits=[],
            reasons=[],
        ),

        human=HumanState(
            decision="",
            decided_at="",
            reviewer="",
            comment="",
        ),

        delivery=DeliveryState(
            diagnosed_at="",
            report_ref="",
            report_uri="",
            traceability=[],
        ),

        workflow=WorkflowState(
            schema_version="1.0",
            degraded_steps=[],
        ),
    )