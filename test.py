"""这个是一个测评rag的代码"""

from backend.ragas_evaluator import evaluate_qa_record
from backend.ark_ragas_llm import ArkRagasLLM

ark_llm = ArkRagasLLM()   # 会自动读取 backend.config 里的 ARK_API_KEY

metrics = evaluate_qa_record(
    question="房颤患者需要做哪些检查？",
    answer="建议 12 导联心电图、BNP/NT-proBNP 等。",
    contexts=[
        "…心电图显示窦性 P 波消失…",
        "…BNP 或 NT-proBNP 显著增高…"
    ],
    ground_truth="12 导联心电图、心肌酶、BNP/NT-proBNP 等",
    llm=ark_llm           # 显式传入，避免 Ragas 自动找 OpenAI
)
print(metrics)