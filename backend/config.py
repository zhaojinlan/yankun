"""backend.config
------------------
集中管理运行时配置。

当前仅包含 ARK_API_KEY：
1. 默认值写在此文件中，方便开箱即用。
2. 若在环境变量中设置同名键，将自动覆盖默认值。
"""
import os
from pathlib import Path

# 默认 Ark API Key，如需更改可修改此处或设置环境变量 ARK_API_KEY
ARK_API_KEY: str = os.getenv("ARK_API_KEY", "2bac91ba-9d5e-403f-8e0a-30a5fce5bde6")

# 本地 Moka-AI m3e-base 嵌入模型路径
EMBED_MODEL_DIR = Path(__file__).resolve().parent / "models" / "models--moka-ai--m3e-base"

__all__ = ["ARK_API_KEY", "EMBED_MODEL_DIR"] 