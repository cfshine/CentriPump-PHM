"""大模型接入层：LLM 客户端实例、结构化输出契约与提示词。

**本模块只关心「怎么把模型调起来」，不关心「拿它的输出做什么」**——校验、清洗、失败
降级在 query_rewrite.py，编排在 manual_agent.py。因此本模块**不 import 本项目任何其他
模块**，是依赖图里最底层的一环：

    manual_agent ─┬─> query_rewrite ──> manual_llm
                  ├─> retriever ──────> knowledge_base
                  └─> manual_llm

为什么单独拆出来：提示词、模型配置、输出契约这三样是**绑死**的——改了 `SubQuestion`
的字段描述就等于改了 prompt 的一部分，换了模型就必须重新审一遍提示词，而 `get_rewriter()`
把三者拼在一起（`prompt | llm.with_structured_output(...)`）。它们又被多个环节共用
（查询重写、后续的检索结果审阅、答案生成），塞在编排层里只会让编排层既当配置中心又当
提示词仓库。

模块分工：

| 职责 | 入口 | 位置 |
| :-- | :-- | :-- |
| **模型实例 / 提示词 / 输出契约** | `get_llm` / `get_rewriter` / `SubQuestion` | 本模块 |
| 重写（原始查询 → 子问题，含校验与降级） | `rewrite_query` / `rewrite_to_list` | query_rewrite.py |
| 检索（query → 片段，含多查询合并） | `search_knowledge_base_multi` | retriever.py |
| 编排（串起上面几步） | `run` | manual_agent.py |

用法：

    from manual_llm import get_llm, get_rewriter, REWRITE_PROMPT

    # 重写链：提示词 + 结构化输出，一次调用拿到 SubQuestion 列表
    output = get_rewriter().invoke({"query": "报警码有哪些？", "history": ""})
    for sub in output.sub_questions:
        print(sub.question, sub.intent, sub.keywords)

    # 裸模型：给后续的「检索结果审阅」「答案生成」用，本模块不管它们怎么用
    print(get_llm().invoke("你好").content)

直接跑本文件可自检配置（未配 key 只打印配置、不联网；配了 key 会真跑一次重写）：

    python manual_llm.py

⚠️ 联网与否取决于是否配置：`get_llm()` 在 `.env` 缺 `LLM_API_KEY`（或 `DEEPSEEK_API_KEY`）
时**明确抛 `ValueError`，不静默降级成假模型**——调用方（`query_rewrite.rewrite_query`）
会把异常转成「退回原查询」的降级结果并打 `[WARN]`，这样问题出在哪一眼可见。
"""

import os

from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

load_dotenv()


# =====================================================================
# 一、配置
# =====================================================================
# 变量名与 `manual_agent/LLM.py` 对齐，方便复用同一份 `.env`：
#   LLM_MODEL / LLM_API_KEY（或 DEEPSEEK_API_KEY）/ LLM_BASE_URL（或 DEEPSEEK_BASE_URL）
# 任何 OpenAI 兼容端点都可用（DeepSeek / 通义 / vLLM / OpenAI 官方…）。
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or ""

# 重写是「改写」不是「创作」，要的是稳定而不是发散，温度调低。
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# 结构化输出的实现方式：
#   "function_calling" —— 走工具调用（OpenAI / DeepSeek 的 deepseek-chat 等，默认）
#   "json_mode"        —— 端点不支持工具调用时改用 JSON 模式。⚠️ 走这条路要求
#                         **渲染后的 prompt 里出现 "json" 字样**（OpenAI 兼容端点的
#                         硬性校验，DeepSeek 会直接 400），SYSTEM_PROMPT 末尾已带上。
#
# ⚠️ 实测（2026-09-30）：`deepseek-flash` 是**思考模式**模型，对 `tool_choice` 直接
# 返回 400 `Thinking mode does not support this tool_choice`——它在 function_calling
# 下**不可用**，要它就得切 json_mode。当前验证可用的是 `deepseek-chat` + function_calling。
STRUCTURED_OUTPUT_METHOD = os.getenv("LLM_STRUCTURED_METHOD", "function_calling")

# 子问题条数：下限 1（原问题本身就很单一，改写成更详细的 1 条即可），
# 上限既写进 prompt 也做代码侧钳制——模型不听话时由代码兜底，不能只靠提示词。
DEFAULT_MAX_SUB_QUESTIONS = 4
HARD_MAX_SUB_QUESTIONS = 8

# 重写的尝试次数上限（含首次）。
#
# ⚠️ **来历要说清楚，否则会误导后来人**：这个重试最初的动机是「重写约 1/4 概率失败」，
# 当时判断为「模型偶发不发起工具调用」的采样问题。**那个判断是错的**——真正的根因是
# `RewriteOutput` 类名原来带前导下划线（`_RewriteOutput`），被端点转义成
# `\_RewriteOutput` 导致解析失败（详见 `RewriteOutput` 的类文档）。改名后实测
# **12/12 成功**，失败率归零。
#
# 那为什么还留着它：现在它是**安全网**，不是修复手段。`None` 返回确实被观测到过
# （网络抖动、端点偶发不按 schema 回都是真实存在的），而有界重试的代价是零——
# 正常路径只调一次（已由 `_test_retry.py` 的边界用例断言）。
#
# 次数取 3：只在**真的失败时**才多两次调用（每次约 1.7s），留一点余量即可。
# 它不为某个已知的高频故障兜底，所以不必调大——真要调大，先测出残留率再说。
DEFAULT_REWRITE_ATTEMPTS = 3


# =====================================================================
# 二、结构化输出契约
# =====================================================================
# 放在本模块而不是 query_rewrite.py 的原因：这两个模型就是「提示词的机器可读形式」，
# 它们与下面的 SYSTEM_PROMPT 必须一起改；而且 `get_rewriter()` 需要 `RewriteOutput`
# 才能拼出链，放在业务模块里会绕成循环依赖。
class SubQuestion(BaseModel):
    """一条子问题。

    `intent` 与 `keywords` 不参与检索，是**给人看的**：重写质量出问题时，靠
    「模型以为自己要问什么」和「它挑了什么词」来定位，比只看一句 question 有用得多。
    """

    question: str = Field(
        description=(
            "一条自洽、详细、可直接拿去做向量检索的子问题。要求："
            "① 单独看就懂，不依赖上下文——所有代词（它 / 这个 / 该泵）必须替换成具体实体"
            "（如「HCP-80-50-200A 离心泵组」「TT_DE 轴承温度测点」）；"
            "② 只问一件事，复合意图要拆开；"
            "③ 用知识库里的术语（IOM / 手册用词），保留型号、位号、数值与单位；"
            "④ 语言与用户提问一致。"
        )
    )
    intent: str = Field(
        description="这条子问题想解决原问题里的哪个意图，一句话说明（如「定位振动超标的原因与排查步骤」）。"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="3~6 个检索关键词，含同义词 / 别名（如「报警代码 Alarm Code」「轴承温度 TT_DE」「机械密封泄漏」）。",
    )


class RewriteOutput(BaseModel):
    """LLM 直接被要求填的结构（只含它该产出的部分）。

    拆成两个模型是因为：`original_query` / `degraded` / `elapsed_ms` 这些字段由代码
    填充，混在一起会让模型有机会去编造它们。

    ⚠️ **类名就是发给 API 的工具名，不要加前导下划线。**（2026-09-30 实测踩坑）
    `with_structured_output` 拿类名当 function/tool 名，而带前导下划线的名字会被
    DeepSeek 端点**转义**成 `\\_RewriteOutput` 发回来，LangChain 解析时报
    `OutputParserException: Unknown tool type: '\\_RewriteOutput'`。

    对照实测（同一批查询各 12 次调用）：

    | 类名 | 失败率 |
    | :-- | :-- |
    | `_RewriteOutput`（原名） | **92%**（1/12 成功） |
    | `RewriteOutput`（现名） | **0%**（12/12 成功） |

    教训：这个名字不是普通的内部标识，它**要过网络**，属于对外契约的一部分。
    改名同理——改了它等于换了一个工具名，模型侧的行为可能跟着变。
    """

    sub_questions: List[SubQuestion] = Field(
        description="拆解出的子问题列表，按重要性从高到低排序；1~4 条，不要硬凑数量。"
    )


class ReviewOutput(BaseModel):
    """LLM 审阅检索结果的输出契约（只含它该产出的部分）。

    与 `RewriteOutput` 同样的拆分理由：`rule_hits` / `llm_consulted` / `elapsed_ms`
    由代码填充，不能给模型编造的机会——那些字段恰恰是排查「这次判断是谁做的」的依据。

    ⚠️ 类名同样即工具名，别加前导下划线（见 `RewriteOutput` 的类文档）。
    """

    sufficient: bool = Field(
        description="现有片段是否足以支撑回答用户的问题。"
        "true = 片段里能直接找到回答所需的关键信息；false = 只是泛泛相关，缺少实质内容。"
    )
    reason: str = Field(description="判断理由，一句话，供人回看误判。")
    missing: List[str] = Field(
        default_factory=list,
        description="sufficient=false 时，说明缺哪部分信息（如「缺少振动报警的具体阈值」）。",
    )
    next_queries: List[str] = Field(
        default_factory=list,
        description="sufficient=false 时，建议用来补查的检索语句，0~3 条。"
        "要针对**缺失的部分**换角度提问，不要重复已有子问题。",
    )


# =====================================================================
# 三、模型实例
# =====================================================================
_llm: Optional[ChatOpenAI] = None
_rewriter: Optional[Runnable] = None
_reviewer: Optional[Runnable] = None


def get_llm() -> ChatOpenAI:
    """通用 LLM 客户端（懒加载单例）。

    **没配置就明确报错**，不静默降级成一个假模型：调用方（`rewrite_query`）会把异常
    转成「退回原查询」的降级结果并打 `[WARN]`，这样问题出在哪一眼可见。

    返回的是**没有绑定任何输出结构**的裸模型，供「检索结果审阅」「答案生成」等环节复用；
    要结构化输出用 `get_rewriter()`。
    """
    global _llm
    if _llm is None:
        if not LLM_API_KEY:
            raise ValueError(
                "未配置大模型：请在 .env 里设置 LLM_API_KEY（或 DEEPSEEK_API_KEY）"
                "与 LLM_MODEL；非 OpenAI 官方端点还要设 LLM_BASE_URL（或 DEEPSEEK_BASE_URL）"
            )
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL or None,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )
    return _llm


# =====================================================================
# 四、查询重写提示词
# =====================================================================
# 规则写得比一般的「请拆解问题」具体得多，是因为重写质量直接决定检索上限：
# 少一条子问题 = 少一个召回方向；一条子问题里塞两个意图 = 该方向被平均掉。
SYSTEM_PROMPT = """你是离心泵运维知识库的「检索查询重写器」。你的唯一任务是把用户的提问改写成一组子问题，\
用于向量检索；你**不回答**这些问题。

知识库内容：HCP 系列卧式单级单吸离心泵组的仿真操作与维修手册、工业离心泵标准操作维护规程（IOM）\
与故障排查知识库——涵盖设备结构与核心测点、启停与操作步骤、报警阈值、故障诊断与处理、维修记录等。

改写规则：
1. **消解指代（按需，不要滥用）**：只把**确实出现**的代词与口语指代替换成具体实体，例如\
「它」「该泵」→「HCP-80-50-200A 离心泵组」；「那个温度报警」→「TT_DE 驱动端轴承温度报警」。\
⚠️ **不要给每条子问题都套上设备型号前缀**：知识库整份都在讲同一台设备，反复写型号既不增加\
区分度，还会把问题本身要问的东西盖过去。原问题没提型号、也没有需要消解的指代时，子问题\
**不要**带型号。
2. **显性化意图**：把模糊说法补成检索意图。例如「泵坏了怎么办」→「离心泵常见故障现象、\
原因与排查处理步骤」；「平时要注意什么」→「离心泵日常巡检项目与维护周期」。
3. **一条只问一件事**：按维度拆开复合问题（设备结构与核心测点 / 启停与操作步骤 / 运行参数与报警阈值 / \
故障诊断与处理 / 维护与维修记录）。不同维度不要塞进同一条。
4. **每条自洽**：单独拿出去检索也必须能被看懂，不依赖原句、不依赖其他子问题。
5. **用知识库的行话**：优先用手册术语；**原问题里出现过的**型号、测点位号（如 TT_DE）、\
报警代码、数值与单位要原样保留，不要改写成同义词；必要时在同一条里带上常见别名或英文，\
提高召回。（注意：是「保留已有的」，不是「补上没提的」——见规则 1。）
6. **不改问题的所指**：你只是把问法写清楚，**不能把问题换成另一个问题**。\
原问题问什么，子问题就问什么。

   ⚠️ **禁止产出「知识库中是否包含 X」这类元问题。** 实测踩坑：把「如何申请报销差旅费」\
改写成「离心泵运维知识库中是否包含差旅费报销流程」，会让它匹配到**知识库的自我介绍**\
（文档开头讲覆盖范围的那一段），余弦相似度从 0.44 抬到 0.64——**本该拒答的问题因此\
通过了拒答闸门**，下游拿到一堆不相干的片段照常作答。

   **原问题明显不属于本知识库范围时（报销、天气、报价、其它型号设备等），照原意改写**，\
让它检索不到东西。**检索不到是正确的信号**，说明该拒答；不要为了「让它能查到」而硬套设备术语。

7. **不编造**：不要写知识库明显不可能有的内容（具体报价、外部工单号等），不要把结论写进子问题。
8. **条数**：1~{max_sub_questions} 条。原问题本身就单一（例如「报警码有哪些」）就只产出 1 条，\
把它写得更详细即可；**不要为凑数量而硬拆**。按重要性从高到低排序。

语言与用户提问保持一致。只以 JSON 输出结构化结果，不要寒暄、不要解释。"""

HUMAN_PROMPT = """{history}用户提问：{query}"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
).partial(max_sub_questions=str(DEFAULT_MAX_SUB_QUESTIONS))


def _structured(schema: type) -> Runnable:
    """`llm.with_structured_output(schema, method=STRUCTURED_OUTPUT_METHOD)`。

    所有结构化链都从这个出口走：`method` 是**端点相关**的配置（见
    `STRUCTURED_OUTPUT_METHOD`），散在几个 `get_*` 里迟早会漏改一处。

    用 `with_structured_output` 而不是「让模型输出 JSON 再自己解析」：前者由
    LangChain 按 schema 生成参数、走工具调用并做校验，模型返回不合法时框架会重试，
    比手写正则解析稳得多。
    """
    return get_llm().with_structured_output(schema, method=STRUCTURED_OUTPUT_METHOD)


def get_rewriter() -> Runnable:
    """重写链（懒加载单例）。输入 `{"query": str, "history": str}`，输出 `RewriteOutput`。"""
    global _rewriter
    if _rewriter is None:
        _rewriter = REWRITE_PROMPT | _structured(RewriteOutput)
    return _rewriter


# =====================================================================
# 五、检索结果审阅提示词
# =====================================================================
# 审阅器与重写器是两种不同的任务，提示词也完全不同：重写要「拆」——把一个模糊问题拆成
# 能各自去检索的子问题；审阅要「判」——看实际召回了什么、够不够，不够就指出缺在哪。
# 两者共同的硬约束是**都不回答用户的问题**，否则模型会拿着不完整的上下文开始写答案。
REVIEW_SYSTEM_PROMPT = """你是离心泵运维知识库的「检索结果审阅器」。你会收到用户提问、为检索而拆解的子问题，\
以及**实际召回到的原文片段**。你的任务是判断这些片段够不够支撑回答；你**不回答**用户的问题。

判断标准：
1. **够**（sufficient=true）：片段里能直接找到回答所需的关键信息——数值、阈值、步骤、判据、处理建议等。
2. **不够**（sufficient=false）：片段只是**泛泛相关**（讲设备构成、章节概述、术语定义），\
没有回答问题所需的实质内容；或者用户问了多个方面，其中某些方面完全没有被任何片段覆盖。
3. **只认片段里写的内容**：片段之外的知识一律不算，不要用「根据常识可以推断」来凑；
   也不要因为片段里出现了相关的词就判为够。

判为不够时，给出 0~3 条**新的检索查询**用来补查：
- 针对**缺失的那部分**提问，不要重复已有的子问题；
- 换一个角度、换用手册里的术语或别名（如「轴承温度」↔「TT_DE」↔「轴承过热」），\
不要只是把原问题重说一遍；
- 语言与用户提问一致。

只以 JSON 输出结构化结果，不要寒暄、不要解释。"""

REVIEW_HUMAN_PROMPT = """用户提问：{query}

为检索而拆解的子问题：
{sub_questions}

待审阅的检索结果：
{retrieval}
"""

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REVIEW_SYSTEM_PROMPT),
        ("human", REVIEW_HUMAN_PROMPT),
    ]
)


def get_reviewer() -> Runnable:
    """审阅链（懒加载单例）。

    输入 `{"query": str, "sub_questions": str, "retrieval": str}`，输出 `ReviewOutput`。
    三个入参都是**已经拼好的文本**：把「检索结果怎么呈现给模型」交给调用方
    （`manual_review.review_retrieval`），本模块不关心检索结果的内部结构。
    """
    global _reviewer
    if _reviewer is None:
        _reviewer = REVIEW_PROMPT | _structured(ReviewOutput)
    return _reviewer


# ⚠️ 已知不足：`REWRITE_PROMPT` 用 `.partial()` 把 `max_sub_questions` 固定成了
# `DEFAULT_MAX_SUB_QUESTIONS`，所以调用方传别的条数时，**提示词里写的仍是默认值**，
# 只有代码侧钳制（`query_rewrite._clean_sub_questions`）在生效。后果是「要得更少」
# 这种请求会被截断而非按需生成。要修就得把该变量改成每次 invoke 时传入。


if __name__ == "__main__":
    print("LLM 配置：")
    print(f"  LLM_MODEL             = {LLM_MODEL}")
    print(f"  LLM_BASE_URL          = {LLM_BASE_URL or '（SDK 默认端点）'}")
    print(f"  LLM_API_KEY           = {'已配置' if LLM_API_KEY else '未配置'}")
    print(f"  LLM_TEMPERATURE       = {LLM_TEMPERATURE}")
    print(f"  LLM_TIMEOUT           = {LLM_TIMEOUT}s")
    print(f"  LLM_STRUCTURED_METHOD = {STRUCTURED_OUTPUT_METHOD}")
    print(
        f"  子问题条数上限         = {DEFAULT_MAX_SUB_QUESTIONS}"
        f"（硬上限 {HARD_MAX_SUB_QUESTIONS}）"
    )

    if not LLM_API_KEY:
        print(
            "\n[WARN] 未配置 LLM_API_KEY（或 DEEPSEEK_API_KEY），get_llm() 会抛 ValueError；"
            "本次不调用模型，请在 .env 中配置后重试。"
        )
    else:
        demo_query = "报警码有哪些？它们各自的阈值是多少？"
        print(f"\n试跑一次重写：{demo_query}\n")
        demo = get_rewriter().invoke({"query": demo_query, "history": ""})
        for index, sub in enumerate(demo.sub_questions, 1):
            print(f"[{index}] {sub.question}")
            if sub.intent:
                print(f"    意图：{sub.intent}")
            if sub.keywords:
                print(f"    关键词：{'、'.join(sub.keywords)}")
