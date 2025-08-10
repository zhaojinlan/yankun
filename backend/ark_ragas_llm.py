"""backend.ark_ragas_llm
-----------------------
将现有 ArkLLM 适配为 Ragas 评估可用的 LLM 接口。
Ragas 0.3.x 里的 LLM 抽象定义 (ragas.llms.base.LLM)：
    • generate(prompts: List[str], temperature: float = 0.0, max_tokens: int = 1024) -> List[str]
    • get_token_count(text: str) -> int

本封装内部复用项目现有的 backend.llm.ArkLLM。
"""
from __future__ import annotations

from typing import List, Any
import asyncio
from dataclasses import dataclass

try:
    from ragas.llms.base import LLM  # type: ignore
except ImportError:  # 兼容新版本 Ragas 无 LLM 基类的情况
    class LLM:  # type: ignore
        """简易占位基类：仅作类型提示，不影响运行"""
        pass
from backend.llm import ArkLLM
from backend.config import ARK_API_KEY


@dataclass
class Generation:
    """单条生成结果的包装器"""
    text: str


@dataclass
class LLMResult:
    """LLM 生成结果的包装器"""
    generations: List[List[Generation]]
    llm_output: dict | None = None


class ArkRagasLLM(LLM):
    """Ragas 兼容的 Ark 大模型封装"""

    def __init__(self, model: str = "deepseek-v3-250324", temperature: float = 0.0, max_tokens: int = 2048):
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # 直接复用已有封装
        self._client = ArkLLM(model=model, api_key=ARK_API_KEY)

    # ---------- Ragas LLM 接口实现 ----------
    async def generate(
        self,
        prompts: List[str],
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,  # Ragas 用这个参数控制每个prompt生成多少个答案
        **kwargs  # 捕获其他可能的参数
    ) -> LLMResult:  # type: ignore[override]
        """批量生成，返回每条 prompt 的结果字符串"""
        temp = temperature if temperature is not None else self._temperature
        max_t = max_tokens if max_tokens is not None else self._max_tokens

        # ArkLLM 当前不支持批量；逐条调用
        all_generations: List[List[Generation]] = []
        for p in prompts:
            prompt_generations: List[Generation] = []
            # 每个prompt可能需要生成多个答案
            for _ in range(n):
                # 在事件循环中运行同步代码
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._client.invoke, p
                )
                prompt_generations.append(Generation(text=result))  # type: ignore[attr-defined]
            all_generations.append(prompt_generations)
        return LLMResult(generations=all_generations)

    def get_token_count(self, text: str) -> int:  # type: ignore[override]
        return len(text.split())

    # ---------- Ragas 新接口兼容 ----------
    def set_run_config(self, run_config):  # type: ignore[override]
        """Ragas 在评估开始时注入运行配置；此处无需特殊处理，仅保存。"""
        self._run_config = run_config

    # ---------- 兼容属性 ----------
    @property
    def model_name(self) -> str:  # for logging/debugging
        return self._model_name 