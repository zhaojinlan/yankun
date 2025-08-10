"""backend.llm
----------------
封装火山方舟 (Ark) 大模型为 LangChain 的 LLM 类。

• ArkLLM 继承自 LangChain LLM 基类，实现 _call 等必要接口。
• 在初始化阶段自动读取 backend.config 中的 ARK_API_KEY （或环境变量）。
• 使业务代码可像使用任何 LangChain 模型一样使用 Ark 大模型。
"""

# noqa: D100, D101, D102  # 停用 pydocstyle 对示例代码的严格检查
import os
# 从统一配置读取默认 Ark Key，便于本地快速运行
from .config import ARK_API_KEY as DEFAULT_ARK_API_KEY
from typing import Any, List, Optional

from pydantic import Field, model_validator
from volcenginesdkarkruntime import Ark

try:
    from langchain_core.language_models.llms import LLM
except ImportError:
    from langchain.llms.base import LLM

class ArkLLM(LLM):
    """火山方舟大模型封装为 LangChain LLM"""

    # 若环境变量未设置，则回退到 backend.config 中的默认值
    api_key: str = Field(default_factory=lambda: os.getenv("ARK_API_KEY", DEFAULT_ARK_API_KEY))
    model: str = "deepseek-v3-250324"
    timeout: int = 1800

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def _check_api_key(cls, values):  # type: ignore[override]
        key = values.get("api_key") if isinstance(values, dict) else None
        key = key or os.getenv("ARK_API_KEY") or DEFAULT_ARK_API_KEY
        if not key:
            raise ValueError("请设置 ARK_API_KEY 环境变量或传入 api_key 参数")
        values["api_key"] = key
        return values

    def __init__(self, **data: Any):
        super().__init__(**data)
        object.__setattr__(self, "_client", Ark(api_key=self.api_key, timeout=self.timeout))

    @property
    def _llm_type(self) -> str:
        return "ark-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None, history: List[dict] = None, **__):  # type: ignore[override]
        messages = history or []
        # 转换角色名称：bot -> assistant
        for msg in messages:
            if msg.get("role") == "bot":
                msg["role"] = "assistant"
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        content: str = response.choices[0].message.content  # type: ignore[assignment]
        if stop:
            for sw in stop:
                if sw in content:
                    content = content.split(sw)[0]
                    break
        return content

    def _get_num_tokens(self, text: str) -> int:
        return len(text.split()) 

    def stream(self, prompt: str, history: List[dict] = None):
        messages = history or []
        # 转换角色名称：bot -> assistant
        for msg in messages:
            if msg.get("role") == "bot":
                msg["role"] = "assistant"
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True              # 关键
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content