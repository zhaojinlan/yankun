"""backend.tools.vector_search
统一封装医学病例向量检索，供其它模块调用。
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np  # noqa: F401  # 可能被外部调用使用
from sentence_transformers import SentenceTransformer

from ..config import EMBED_MODEL_DIR

# ---------------- 路径常量 ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
VECTOR_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DIR / "medical.index"
META_PATH = VECTOR_DIR / "metadata.pkl"

# ---- 私有全局缓存 ----
_embed_model: Optional[SentenceTransformer] = None
_vector_index: Optional[faiss.Index] = None
_vector_metadata: Optional[List[dict]] = None


def _load_embed_model() -> SentenceTransformer:
    """加载（或缓存）嵌入模型"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    snapshot_dir = EMBED_MODEL_DIR / "snapshots"
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.glob("*"))
        model_path = snapshots[0] if snapshots else EMBED_MODEL_DIR
    else:
        model_path = EMBED_MODEL_DIR

    if model_path.exists():
        relative_path = model_path.relative_to(PROJECT_ROOT)
        print(f"✅ 使用本地嵌入模型: {relative_path}")
        _embed_model = SentenceTransformer(str(model_path))
    else:
        print("⚠️ 本地嵌入模型不存在，将在线下载: moka-ai/m3e-base")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
        _embed_model = SentenceTransformer("moka-ai/m3e-base")
    return _embed_model


def _load_vector_db():
    """加载（或缓存）Faiss 向量库及元数据"""
    global _vector_index, _vector_metadata
    if _vector_index is not None and _vector_metadata is not None:
        return _vector_index, _vector_metadata

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "未找到 vector_db，请先运行 json_vectorizer.py 生成索引\n"
            "注意：json_vectorizer.py 需确保添加 type='symptom_case' 字段"
        )
    _vector_index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        _vector_metadata = pickle.load(f)
    print(f"📊 已加载向量库: {len(_vector_metadata)} 条医学知识条目")
    return _vector_index, _vector_metadata


class MedicalCaseRetriever:
    """医学病例向量检索器（带内部缓存）"""

    def __init__(self):
        self.embed = _load_embed_model()
        self.index, self.metadata = _load_vector_db()

    def search_cases(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        q_vec = self.embed.encode(query).astype("float32").reshape(1, -1)
        # 检索更多候选以覆盖
        enlarge_k = top_k * 15
        scores, indices = self.index.search(q_vec, enlarge_k)

        results: List[Tuple[float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if meta.get("type") != "symptom_case":
                continue
            if "full_data" not in meta or not isinstance(meta["full_data"], dict):
                continue
            results.append((float(score), meta))
            if len(results) >= top_k:
                break

        if not results:
            print("⚠️ 未匹配到医学知识，请检查 json_vectorizer.py 是否正确添加 type='symptom_case'")
        return results


# -------- 模块级便捷函数（单例） --------
_default_retriever: Optional[MedicalCaseRetriever] = None


def get_retriever() -> MedicalCaseRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = MedicalCaseRetriever()
    return _default_retriever


def search_cases(query: str, top_k: int = 3):
    """对外统一入口"""
    return get_retriever().search_cases(query, top_k)


__all__ = ["MedicalCaseRetriever", "search_cases", "get_retriever"] 