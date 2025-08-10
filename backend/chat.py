"""backend.chat
----------------
封装对话相关功能，供 FastAPI 路由或其他业务代码调用。

目前提供四个主要接口：

1. assistant_reply  —— 普通助手角色的一次性回复
2. patient_reply    —— 病患/病患家属角色的一次性回复
3. assistant_stream —— 普通助手角色的流式回复生成器
4. patient_stream   —— 病患/病患家属角色的流式回复生成器

示例：

>>> from backend.chat import assistant_reply
>>> assistant_reply("你好！", [])
"您好，请问有什么可以帮助您的？"

FastAPI 路由可直接引用这些函数，简化 server.py 中的实现。
"""

from __future__ import annotations

from typing import Dict, Generator, List

# 复用 ArkLLM
from .llm import ArkLLM

__all__ = [
    "assistant_reply",
    "patient_reply",
    "scenario_patient_reply",
    "scenario_patient_stream",
    "assistant_stream",
    "patient_stream",
    "SCENARIO_PROMPTS",
    "build_history",
]


# =========================== 内部工具与配置 ===========================


_SYSTEM_PROMPT = "你是一名乐于助人的 AI 助手，请用中文简洁回答用户的问题。"

# 可以根据需要自定义病情、身份、情绪等
_PATIENT_PROMPT = "你是一名病患（或病患家属），正在向医生描述症状、表达顾虑并询问治疗方案。"

# 场景特定提示词，供医学实习生模拟不同沟通情景
SCENARIO_PROMPTS = {
    "emergency": (
        "你现在扮演【急诊患者的家属】并用第一人称回答，处于情绪紧张状态，需要从医生处了解患者当前状况，并希望尽快采取措施。请避免使用医学专业术语，表达关切、疑惑与情绪。"
    ),
    "bad_news": (
        "你现在扮演【病情恶化患者的家属】并用第一人称回答，刚接到坏消息，情绪低落且焦虑。与医生沟通时，请表达震惊、悲伤，提出疑问，并寻求后续治疗与安慰。避免使用医学专业术语。"
    ),
    "notification": (
        "你现在扮演【需要转院/转科患者的家属】并用第一人称回答，对转院决定感到疑惑，希望医生说明原因及方案。避免使用医学专业术语，以普通人语气表达关切。"
    ),
    "preoperation": (
        "你现在扮演【即将手术患者的家属】并用第一人称回答，想了解手术风险、术前准备与可能并发症，并确认知情同意。避免医学术语，以亲属身份提问。"
    ),
}


_llm: ArkLLM | None = None


def _get_llm() -> ArkLLM:  # pragma: no cover
    """延迟初始化 ArkLLM，避免重复创建客户端。"""

    global _llm
    if _llm is None:
        _llm = ArkLLM()
    return _llm


def build_history(system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """在已有 history 前插入 system_prompt。"""

    return [{"role": "system", "content": system_prompt}, *history]


# =========================== 同步回复接口 ===========================


def assistant_reply(message: str, history: List[Dict[str, str]] | None = None) -> str:
    """助手角色的一次性回复。"""

    history = history or []
    messages = build_history(_SYSTEM_PROMPT, history) + [{"role": "user", "content": message}]
    llm = _get_llm()
    resp = llm._client.chat.completions.create(model=llm.model, messages=messages)
    return resp.choices[0].message.content  # type: ignore[return-value]


def patient_reply(message: str, history: List[Dict[str, str]] | None = None) -> str:
    """病患/家属角色的一次性回复。"""

    history = history or []
    messages = build_history(_PATIENT_PROMPT, history) + [{"role": "user", "content": message}]
    llm = _get_llm()
    resp = llm._client.chat.completions.create(model=llm.model, messages=messages)
    return resp.choices[0].message.content  # type: ignore[return-value]


# =========================== 场景化回复接口 ===========================


def scenario_patient_reply(
    message: str,
    scenario: str,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """根据场景模拟病患/家属一次性回复。

    参数
    ------
    message: 用户输入
    scenario: 场景标识符，必须在 SCENARIO_PROMPTS 中
    history: 对话历史
    """

    if scenario not in SCENARIO_PROMPTS:
        raise ValueError(f"未知场景: {scenario}. 可用场景: {list(SCENARIO_PROMPTS)}")

    prompt = SCENARIO_PROMPTS[scenario]
    history = history or []
    messages = build_history(prompt, history) + [{"role": "user", "content": message}]
    llm = _get_llm()
    resp = llm._client.chat.completions.create(model=llm.model, messages=messages)
    return resp.choices[0].message.content  # type: ignore[return-value]


def scenario_patient_stream(
    message: str,
    scenario: str,
    history: List[Dict[str, str]] | None = None,
):
    """根据场景模拟病患/家属流式回复生成器。"""

    if scenario not in SCENARIO_PROMPTS:
        raise ValueError(f"未知场景: {scenario}. 可用场景: {list(SCENARIO_PROMPTS)}")

    prompt = SCENARIO_PROMPTS[scenario]
    history = history or []
    base_history = build_history(prompt, history)
    llm = _get_llm()
    for token in llm.stream(message, history=base_history):
        yield token


# =========================== 流式回复接口 ===========================


def assistant_stream(message: str, history: List[Dict[str, str]] | None = None) -> Generator[str, None, None]:
    """助手角色流式回复生成器。"""

    history = history or []
    base_history = build_history(_SYSTEM_PROMPT, history)
    llm = _get_llm()
    for token in llm.stream(message, history=base_history):
        yield token


def patient_stream(message: str, history: List[Dict[str, str]] | None = None) -> Generator[str, None, None]:
    """病患/家属角色流式回复生成器。"""

    history = history or []
    base_history = build_history(_PATIENT_PROMPT, history)
    llm = _get_llm()
    for token in llm.stream(message, history=base_history):
        yield token
