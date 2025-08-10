from __future__ import annotations

"""backend.ragas_evaluator
--------------------------------
提供使用 Ragas 对单条 QA 记录进行自动评估的便捷函数。

示例：
    from backend.ragas_evaluator import evaluate_qa_record

    metrics = evaluate_qa_record(
        question="患者出现心悸、胸闷，需要做哪些检查？",
        answer="建议 12 导联心电图、BNP/NT-proBNP 等进一步评估……",
        contexts=[
            "…心电图显示窦性 P 波消失…",
            "…BNP 或 NT-proBNP 显著增高…",
        ],
        ground_truth="12 导联心电图、心肌酶、BNP/NT-proBNP 等"
    )
    print(metrics)

注意：
    1. 需要在环境变量中设置 OPENAI_API_KEY 或自行配置 ragas.llms。
    2. 为控制成本，建议仅在调试或离线评估数据集时调用。
"""

from typing import List, Dict, Optional
from typing import TYPE_CHECKING

from ragas import evaluate
from ragas.evaluation import Dataset
import pandas as pd
import os
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # 兼容旧版本

# 若环境中未配置 OPENAI_API_KEY，则自动使用 Ark LLM
_USE_ARK_AS_DEFAULT = not os.getenv("OPENAI_API_KEY")
if _USE_ARK_AS_DEFAULT:
    try:
        from backend.ark_ragas_llm import ArkRagasLLM  # noqa
        _DEFAULT_LLM = ArkRagasLLM()
    except Exception as _e:  # pylint: disable=bare-except
        _DEFAULT_LLM = None
else:
    _DEFAULT_LLM = None

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


_DEFAULT_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


if TYPE_CHECKING:
    try:
        from ragas.llms.base import LLM  # type: ignore
    except ImportError:
        from typing import Protocol

        class LLM(Protocol):  # type: ignore
            def generate(self, prompts: list[str], temperature: float = 0.0, max_tokens: int = 1024) -> list[str]: ...
            def get_token_count(self, text: str) -> int: ...
    
    try:
        from langchain_core.embeddings import Embeddings  # type: ignore
    except ImportError:
        from typing import Protocol
        
        class Embeddings(Protocol):  # type: ignore
            def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
            def embed_query(self, text: str) -> list[float]: ...


def evaluate_qa_record(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    metrics: Optional[list] = None,
    llm: Optional["LLM"] = None,
    embeddings: Optional["Embeddings"] = None,
) -> Dict[str, float]:
    """使用 Ragas 评估单条问答记录。

    参数
    ------
    question: 提问内容
    answer:   系统生成的答案
    contexts: 检索到的知识片段（字符串列表）
    ground_truth: (可选) 人工标注的期望答案，用于计算 correctness 等需 GT 的指标
    metrics: (可选) 传入自定义指标列表，默认使用 4 项常用指标

    返回
    ------
    dict: {metric_name: score}
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    # 构建 DataFrame
    df = pd.DataFrame([{
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "ground_truth": ground_truth if ground_truth is not None else ""
    }])
    print("Dataset columns:", df.columns.tolist())  # 调试信息
    print("First row:", df.iloc[0].to_dict())      # 调试信息

    # 准备评估参数
    kwargs = {"metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm  # 允许自定义评估模型
    elif _DEFAULT_LLM is not None:
        kwargs["llm"] = _DEFAULT_LLM
    
    # 处理嵌入模型
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    elif not os.getenv("OPENAI_API_KEY"):
        # 若无 OpenAI Key，使用本地嵌入模型
        try:
            from backend.config import EMBED_MODEL_DIR
            # 找到具体的snapshot目录
            snapshot_dir = next((EMBED_MODEL_DIR / "snapshots").glob("*"))
            kwargs["embeddings"] = HuggingFaceEmbeddings(
                model_name=str(snapshot_dir),
                cache_folder=str(EMBED_MODEL_DIR.parent),
                model_kwargs={"device": "cpu"}  # 避免CUDA相关警告
            )
        except Exception:
            # 备用：在线下载通用嵌入模型
            kwargs["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 从 DataFrame 构建 Dataset
    dataset = Dataset.from_pandas(df)
    results = evaluate(dataset, **kwargs)

    # DataFrame → dict，字段名即指标名称
    df = results.to_pandas()
    return df.iloc[0].to_dict() 