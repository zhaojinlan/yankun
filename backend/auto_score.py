import json
from typing import Any, Dict, List

from .llm import ArkLLM


def _parse_score_json(text: str) -> Dict[str, Any]:
    """从模型返回文本中解析评分 JSON。

    期望格式：
    {
      "score": number,           # 0-10，允许一位小数
      "reason": "...",          # 简要说明
      "reviews": [               # 可选，多评审合议
        {"critic": "心理学家", "score": number, "reason": "..."},
        {"critic": "语言学家", "score": number, "reason": "..."}
      ]
    }
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            # 兜底字段
            if not isinstance(obj.get("reviews", []), list):
                obj["reviews"] = []
            return {
                "score": float(obj.get("score", 0.0)),
                "reason": str(obj.get("reason", "")),
                "reviews": obj.get("reviews", []),
            }
    except Exception:
        pass
    # 解析失败兜底
    return {"score": 0.0, "reason": "评分解析失败", "reviews": []}


def get_score(question: str, answer: str) -> Dict[str, Any]:
    """对一问一答进行 AI 主观评分。

    返回字典包含：{"score": float, "reason": str, "reviews": List[...]}。
    """
    llm = ArkLLM()

    system = (
        "你是一名医疗沟通评估专家，精通心理学与语言学，能够对医生回复进行结构化评分。"
        "评分标准：0-10 分，允许一位小数，越高代表越优。"
        "仅输出 JSON，不要任何额外文字。"
    )

    prompt = f"""
请根据对话环境，对医生的回答进行评分，并给出理由与多评审视角。

【对话】
- 家属提问（其他输入）：{question}
- 医生回答（用户输入）：{answer}

【输出要求】严格输出一个 JSON 对象：
{{
  "score": <0-10 的数字，允许一位小数>,
  "reason": "简要说明评分的核心依据（不超过80字）",
  "reviews": [
    {{"critic": "心理学家", "score": <数字>, "reason": "从情感与共情角度简述"}},
    {{"critic": "语言学家", "score": <数字>, "reason": "从表达清晰与结构角度简述"}}
  ]
}}
仅输出 JSON，不要任何解释。
"""

    text = llm._call(prompt=prompt, history=[{"role": "system", "content": system}])
    result = _parse_score_json(text)

    # 结果规范化与裁剪
    score = result.get("score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0
    score = max(0.0, min(10.0, float(score)))

    reason = str(result.get("reason", "")).strip()
    reviews: List[Dict[str, Any]] = result.get("reviews", []) or []

    # 归一化 reviews
    normalized_reviews: List[Dict[str, Any]] = []
    for r in reviews:
        try:
            normalized_reviews.append({
                "critic": str(r.get("critic", "评审")).strip() or "评审",
                "score": max(0.0, min(10.0, float(r.get("score", score)))) ,
                "reason": str(r.get("reason", "")).strip(),
            })
        except Exception:
            continue

    return {"score": score, "reason": reason, "reviews": normalized_reviews}


if __name__ == "__main__":
    # 简单命令行演示
    q = "我的母亲怎么了医生，她的病很严重吗？"
    a = "您好，我非常理解您的感受，请您不要担心，您的母亲正在接受治疗，病情正在好转。"
    print(get_score(q, a))