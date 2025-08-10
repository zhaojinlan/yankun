#!/usr/bin/env python3
"""CLI 快捷脚本：
$ python ask.py "患者主诉剧烈胸痛，伴大汗" --top_k 5
内部调用 backend.diagnosis_agent.get_diagnosis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证可导入 backend 包
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import get_diagnosis  # noqa: E402


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="向量检索 + 大模型诊疗建议")
    parser.add_argument("query", help="患者主诉或症状描述")
    parser.add_argument("--top_k", type=int, default=3, help="返回相似病例数")
    args = parser.parse_args()

    answer = get_diagnosis(args.query, top_k=args.top_k)
    print("\n======== 诊疗建议 ========")
    print(answer)


if __name__ == "__main__":
    main()
