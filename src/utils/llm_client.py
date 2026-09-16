# src/utils/llm_client.py
"""大模型客户端。

走真实 DeepSeek（OpenAI 兼容）。凭据由 langchain 自动从系统环境变量
`DEEPSEEK_API_KEY` 读取（`model_provider="deepseek"` 的默认行为），
本文件不做任何凭据的显式声明或获取。
"""
from langchain.chat_models import init_chat_model

#: 模型单例。构造时 init_chat_model 会自动去环境变量里找 DEEPSEEK_API_KEY。
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    max_tokens=4096,
    # ★ 工业诊断要求**同一窗口、同一数据、同一结论**（可复现 =
    #   报告能溯源、测试不 flaky）。0.7 会让同一窗口每次描述长度在 765~1031 字
    #   之间跳动，字数断言必然时好时坏。改 0 后同样输入得到同样输出。
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
    )
