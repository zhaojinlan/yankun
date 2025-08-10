"""backend.tools.json_vectorizer
---------------------------------
将结构化病例 JSON 转换为向量并保存到项目根目录下的 vector_db/。
运行示例：
   python -m backend.tools.json_vectorizer backend/json_db --append
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os, pickle
from typing import List, Tuple
from pathlib import Path
from backend.config import EMBED_MODEL_DIR

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 新增: 统一获取疾病列表的工具函数，兼容不同 JSON 结构
def _get_diseases_from_data(data: dict) -> list:
    """返回一个疾病字典列表，兼容以下两种结构:  
    1. data['possible_diseases'] -> List[dict]  
    2. data['disease'] -> dict
    """
    if 'possible_diseases' in data and isinstance(data['possible_diseases'], list):
        return data['possible_diseases']
    elif 'disease' in data and isinstance(data['disease'], dict):
        return [data['disease']]
    # 默认返回空列表，调用方自行处理
    return []

class JSONVectorizer:
    def __init__(self):
        # 如果本地模型目录存在则优先使用，否则回退在线模型名
        if EMBED_MODEL_DIR.exists():
            # snapshots 结构处理
            snapshots = list((EMBED_MODEL_DIR / "snapshots").glob("*"))
            model_path = snapshots[0] if snapshots else EMBED_MODEL_DIR
            print(f"✅ 使用本地嵌入模型: {model_path}")
            self.model = SentenceTransformer(str(model_path))
        else:
            print("⚠️ 本地模型不存在，将在线加载 'sentence-transformers/all-MiniLM-L6-v2'")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def extract_features(self, data):
        """提取JSON特征 (兼容 possible_diseases 与 disease 两种结构)"""
        features = {}

        # 症状特征
        features['symptom'] = f"{data.get('symptom', '')} {data.get('symptom_category', '')} {' '.join(data.get('severity_indicators', []))}"

        # 疾病特征
        disease_texts = []
        diseases = _get_diseases_from_data(data)
        for disease in diseases:
            disease_name = disease.get('disease') or disease.get('name', '')
            probability = disease.get('probability', '')
            urgency = disease.get('urgency_level', '')
            key_features = disease.get('key_features', [])
            text = f"{disease_name} 概率:{probability} 紧急:{urgency} {' '.join(key_features)}"
            disease_texts.append(text)
        features['diseases'] = disease_texts

        # 检查特征
        test_texts = []
        for disease in diseases:
            for test in disease.get('recommended_tests', []):
                test_name = test.get('test') or test.get('test_name', '')
                purpose = test.get('purpose', '')
                positive_indicators = test.get('positive_indicators', [])
                text = f"{test_name} {purpose} {' '.join(positive_indicators)}"
                test_texts.append(text)
        features['tests'] = test_texts

        return features
    
    def vectorize(self, data):
        """向量化JSON数据 - 每个字段单独向量化"""
        features = self.extract_features(data)
        
        # 1. 症状向量（主要检索目标）
        symptom_vector = self.model.encode(features['symptom'])
        symptom_vector = symptom_vector / np.linalg.norm(symptom_vector)  # L2归一化
        
        # 2. 疾病向量（每个疾病一个向量）
        disease_vectors = []
        for text in features['diseases']:
            vec = self.model.encode(text)
            vec = vec / np.linalg.norm(vec)
            disease_vectors.append(vec)
        
        # 3. 检查向量（每个检查一个向量）
        test_vectors = []
        for text in features['tests']:
            vec = self.model.encode(text)
            vec = vec / np.linalg.norm(vec)
            test_vectors.append(vec)
        
        return {
            'symptom_vector': symptom_vector,
            'disease_vectors': disease_vectors,
            'test_vectors': test_vectors
        }

def build_and_save_index(json_files: List[str], output_dir: str | None = None,
                       append: bool = False):
    """
    根据给定的 JSON 文件列表构建 FAISS 向量索引，并将索引与元数据保存到磁盘。
    每个 JSON 文件会生成多个向量条目：
    - 1个症状向量
    - N个疾病向量（每个疾病一个）
    - M个检查向量（每个检查一个）
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "vector_db")
    os.makedirs(output_dir, exist_ok=True)
    vectorizer = JSONVectorizer()

    vectors: List[np.ndarray] = []  # 新增向量
    metadata: List[dict] = []      # 新增元数据

    for path in json_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vecs = vectorizer.vectorize(data)
        
        # 1. 添加症状向量
        vectors.append(vecs['symptom_vector'])
        metadata.append({
            'file_path': path,
            'type': 'symptom_case',
            'content': data.get('symptom', ''),
            'category': data.get('symptom_category', ''),
            'full_data': data
        })
        
        # 2. 添加疾病向量
        diseases = _get_diseases_from_data(data)
        for i, disease_vec in enumerate(vecs['disease_vectors']):
            vectors.append(disease_vec)
            disease_name = diseases[i].get('disease') or diseases[i].get('name', '')
            metadata.append({
                'file_path': path,
                'type': 'disease',
                'content': disease_name,
                'category': data.get('symptom_category', ''),
                'original_data': data
            })
        
        # 3. 添加检查向量
        test_idx = 0
        for disease in diseases:
            for test in disease.get('recommended_tests', []):
                if test_idx < len(vecs['test_vectors']):
                    vectors.append(vecs['test_vectors'][test_idx])
                    test_name = test.get('test') or test.get('test_name', '')
                    metadata.append({
                        'file_path': path,
                        'type': 'test',
                        'content': test_name,
                        'category': data.get('symptom_category', ''),
                        'original_data': data
                    })
                    test_idx += 1
                else:
                    print(f"⚠️ 检查向量数量不足: 需要 {test_idx + 1} 个，但只有 {len(vecs['test_vectors'])} 个")

    if not vectors:
        raise ValueError("未提供任何 JSON 文件，无法构建索引！")

    dim = len(vectors[0])

    index_path = os.path.join(output_dir, 'medical.index')
    meta_path = os.path.join(output_dir, 'metadata.pkl')

    if append and os.path.exists(index_path) and os.path.exists(meta_path):
        # 追加模式：加载现有索引与元数据
        index = faiss.read_index(index_path)
        if index.d != dim:
            raise ValueError(f"向量维度不匹配: 现有 {index.d}, 新数据 {dim}")
        with open(meta_path, 'rb') as f:
            old_metadata: List[dict] = pickle.load(f)
        index.add(np.array(vectors).astype('float32'))
        metadata = old_metadata + metadata
    else:
        # 全量重建
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(vectors).astype('float32'))

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✅ 向量索引已保存到 {index_path}")
    print(f"✅ 元数据已保存到 {meta_path}")
    print(f"📊 总共 {len(vectors)} 个向量条目")
    print(f"📁 来自 {len(json_files)} 个 JSON 文件")


# ---------------- CLI ----------------

def _cli() -> None:
    """命令行入口: 支持通配符、目录参数，长期可用。"""
    import argparse, glob, os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="批量向量化 JSON 并生成索引")
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON 文件或目录, 支持通配符 (如 data/*.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="索引输出目录, 默认 vector_db/",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加模式: 保留已有向量库并插入新数据",
    )
    args = parser.parse_args()

    # 收集 JSON 文件
    json_files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            json_files.extend(glob.glob(os.path.join(p, "*.json")))
        else:
            json_files.extend(glob.glob(p))

    json_files = [str(Path(f)) for f in json_files if f.lower().endswith(".json")]
    if not json_files:
        parser.error("未找到任何 JSON 文件！")

    build_and_save_index(json_files, output_dir=args.output, append=args.append)


if __name__ == "__main__":
    _cli() 