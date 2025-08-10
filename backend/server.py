from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
# 流式响应与异步工具
from starlette.responses import StreamingResponse
import asyncio
# ---------------- 外部业务逻辑 ----------------
from backend.diagnosis import get_diagnosis
# ---------------- 对话封装 ----------------
from .chat import (
    assistant_reply,
    assistant_stream,
    patient_reply,
    patient_stream,
    scenario_patient_reply,
    scenario_patient_stream,
    SCENARIO_PROMPTS,
)
# 引入 Ark LLM 用于诊断等其他模块，如需
from .llm import ArkLLM
from .config import ARK_API_KEY
from .confirm import analyze_medical_case, confirm_disease_case
from .ragas_evaluator import evaluate_qa_record
from .auto_score import get_score as autogen_score

app = FastAPI(title="急诊智能诊断 API", version="1.0.0")

# ------------------ 通用 LLM ------------------
# 全局初始化，避免重复创建客户端
# 延迟初始化，避免启动时验证失败
_llm = None

def _get_llm():
    """获取 LLM 实例，延迟初始化"""
    global _llm
    if _llm is None:
        _llm = ArkLLM()
    return _llm

# 提示词已经在 backend.chat 中定义，无需重复

# 允许前端本地调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnosisRequest(BaseModel):
    query: str
    history: list = []
    top_k: int = 5


class DiagnosisResponse(BaseModel):
    result: str


# -------- 对话（Chat） ---------


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  # [{role: 'user'|'assistant', content: str}, ...]


class ChatResponse(BaseModel):
    reply: str


# 专用响应模型，保持与 ChatResponse 一致，便于前端复用
class PatientChatResponse(BaseModel):
    reply: str
    score: float = 0.0
    reason: str = ""
    reviews: list = []


class ScenarioChatRequest(BaseModel):
    message: str
    scenario: str  # emergency|bad_news|notification|preoperation 之一
    history: List[Dict[str, str]] = []


class ScenarioChatResponse(BaseModel):
    reply: str


def _build_history(system_prompt: str, history: List[Dict[str, str]]):
    """在历史消息前添加系统指令。"""
    return ([{"role": "system", "content": system_prompt}] + list(history))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """普通（一次性）对话回复
    
    使用 ArkLLM 进行单轮对话，支持历史消息上下文。
    
    请求参数:
        - message: 用户输入的消息
        - history: 对话历史记录，格式为 [{"role": "user|assistant", "content": "消息内容"}]
    
    返回:
        - reply: AI 助手的回复内容
    """
    content = assistant_reply(req.message, history=req.history)
    return {"reply": content}


# =============================  Patient Chat  =============================


@app.post("/api/patient_chat", response_model=PatientChatResponse)
async def patient_chat(req: ChatRequest):
    """模拟病患/病患家属角色的对话。

    与 /api/chat 接口实现基本一致，但使用 _PATIENT_PROMPT 构造系统角色，
    使大模型以病患或家属的身份回应。
    """
    # 若医生尚未提问（message 为空），由模型先输出第一句话，不进行评分
    if not req.message.strip():
        content = patient_reply("请先用一句问候向医生开场，并简要描述你的主要症状或诉求。", history=req.history)
        return {
            "reply": content,
            "score": 0.0,
            "reason": "",
            "reviews": [],
        }

    # 医生已给出提问：同步生成患者回复并进行多智能体评分
    content = patient_reply(req.message, history=req.history)
    score_result = autogen_score(req.message, content)
    return {
        "reply": content,
        "score": score_result.get("score", 0.0),
        "reason": score_result.get("reason", ""),
        "reviews": score_result.get("reviews", []),
    }


# ---- 流式版本 ----


@app.post("/api/patient_chat/stream")
async def patient_chat_stream(req: ChatRequest):
    """模拟病患/家属角色的流式对话。"""

    def token_generator():
        for token in patient_stream(req.message, history=req.history):
            yield token.encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


# =============================  AutoGen (moni.py) Chat  =============================

# 导入重构后的 moni 模块





# ---- 流式对话 ----
# 重复导入已上移，移除


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式输出，对应前端实时显示
    
    使用 ArkLLM 进行流式对话，支持历史消息上下文。
    
    请求参数:
        - message: 用户输入的消息
        - history: 对话历史记录
    
    返回:
        - 流式文本数据，逐 token 发送
    
    特点:
        - 基于 ArkLLM 的流式输出
        - 支持对话历史上下文
        - 实时 token 级别输出
    """

    def token_generator():
        # ArkLLM.stream 会自动在末尾加入用户消息
        for token in assistant_stream(req.message, history=req.history):
            yield token.encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


@app.post("/api/scenario_chat", response_model=ScenarioChatResponse)
async def scenario_chat(req: ScenarioChatRequest):
    """根据场景模拟病患/家属的一次性对话。

    若 `message` 为空，则视为会话首次进入，由模型主动发起开场白。
    """

    if not req.message.strip():
        opening = "请先用一句问候向医生开场，并表达你的主要诉求或症状。"
        content = scenario_patient_reply(opening, scenario=req.scenario, history=req.history)
    else:
        content = scenario_patient_reply(req.message, scenario=req.scenario, history=req.history)

    return {"reply": content}


@app.post("/api/scenario_chat/stream")
async def scenario_chat_stream(req: ScenarioChatRequest):
    """场景化流式对话接口。

    若 `message` 为空，则模型先主动输出第一句话。
    """

    def token_generator():
        init_msg = req.message.strip()
        if not init_msg:
            init_msg = "请先用一句问候向医生开场，并表达你的主要诉求或症状。"

        for token in scenario_patient_stream(init_msg, scenario=req.scenario, history=req.history):
            yield token.encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


@app.post("/api/diagnosis", response_model=DiagnosisResponse)
async def diagnose(req: DiagnosisRequest):
    """智能医疗诊断接口
    
    基于向量检索和 RAG 技术的医疗诊断系统，从医学指南中检索相关证据。
    
    请求参数:
        - query: 患者主诉或症状描述
        - history: 对话历史记录
        - top_k: 返回相似病例数量，默认 5
    
    返回:
        - result: 包含诊断建议、检查建议、治疗方案的完整报告
    
    技术特点:
        - 向量检索：从医学知识库中检索相关病例
        - RAG 技术：结合检索结果生成诊断建议
        - 证据溯源：提供诊断依据和参考来源
    """
    result = get_diagnosis(req.query, history=req.history, top_k=req.top_k)
    return {"result": result}


class AnalysisRequest(BaseModel):
    history: List[Dict[str, str]]
    test_results: str


@app.post("/api/analyze")
async def analyze_case(req: AnalysisRequest):
    """智能体分析：结合对话历史和检查结果
    
    根据对话历史和最新检查结果，对病情进行再评估和更新诊断。
    
    请求参数:
        - history: 对话历史记录
        - test_results: 最新检查结果（自由文本）
    
    返回:
        - updated_diagnosis: 更新后的诊断结论
        - confidence_change: 置信度变化（如 +12% 或 -5%）
        - next_steps: 建议下一步操作列表
        - urgent_actions: 必须立即执行的动作列表
        - new_evidence: 新的检查证据列表
    
    应用场景:
        - 检查结果出来后更新诊断
        - 病情变化时的重新评估
        - 治疗方案调整的依据
    """
    result = analyze_medical_case(req.history, req.test_results)
    return {
        "updated_diagnosis": result.updated_diagnosis,
        "confidence_change": result.confidence_change,
        "next_steps": result.next_steps,
        "urgent_actions": result.urgent_actions,
        "new_evidence": [
            {
                "test_name": r.test_name,
                "result_value": r.result_value,
                "unit": r.unit,
                "is_abnormal": r.is_abnormal
            }
            for r in result.new_evidence
        ]
    }


# -------- 确诊 ---------

class ConfirmRequest(BaseModel):
    history: List[Dict[str, str]] = []
    diagnosis_md: str | None = ""
    test_results: str


class ConfirmResponse(BaseModel):
    result: str


@app.post("/api/confirm", response_model=ConfirmResponse)
async def confirm_disease(req: ConfirmRequest):
    """根据首选检查结果给出确诊意见
    
    基于初步诊断和检查结果，给出最终确诊结论和治疗方案。
    
    请求参数:
        - history: 对话历史记录
        - diagnosis_md: 初步诊断（Markdown 格式）
        - test_results: 检查结果
    
    返回:
        - result: 最终确诊报告（Markdown 格式）
    
    报告内容:
        - 明确疾病名称和严重程度
        - 确诊依据
        - 治疗方案
        - 禁忌事项
        - 预后评估
    """
    diag_md = req.diagnosis_md
    if not diag_md:
        # 若前端未明确传入，取最后一条 assistant 消息
        for msg in reversed(req.history):
            if msg.get("role") == "assistant":
                diag_md = msg.get("content", "")
                break
    result_md = confirm_disease_case(diag_md, req.test_results)
    return {"result": result_md}


# -------- 流式输出版本 ---------
# 重复导入已上移，移除


@app.post("/api/diagnosis/stream")
async def diagnose_stream(req: DiagnosisRequest):
    """智能医疗诊断流式接口
    
    诊断结果的流式输出版本，逐行返回诊断报告。
    
    请求参数:
        - query: 患者主诉或症状描述
        - history: 对话历史记录
        - top_k: 返回相似病例数量
    
    返回:
        - 流式文本数据，按行分块发送
    
    特点:
        - 实时显示诊断生成过程
        - 支持长报告的分段显示
        - 提升用户体验
    """
    result = get_diagnosis(req.query, history=req.history, top_k=req.top_k)

    async def line_generator():
        for line in result.split("\n"):
            yield (line + "\n").encode("utf-8")
            # 小延迟保证前端逐段收到
            await asyncio.sleep(0)

    return StreamingResponse(line_generator(), media_type="text/plain; charset=utf-8")


# -------- 结果评估（Ragas） ---------
class RagasRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str | None = None

class RagasResponse(BaseModel):
    metrics: Dict[str, float]


@app.post("/api/evaluate", response_model=RagasResponse)
async def evaluate_answer(req: RagasRequest):
    """使用 Ragas 对单条问答进行自动评估
    
    评估诊断建议的质量，包括相关性、准确性、完整性等指标。
    
    请求参数:
        - question: 提问内容
        - answer: 系统生成的答案
        - contexts: 检索到的知识片段列表
        - ground_truth: 人工标注的期望答案（可选）
    
    返回:
        - metrics: 评估指标字典
          - faithfulness: 答案对检索内容的忠实度
          - answer_relevancy: 答案与问题的相关性
          - context_precision: 检索内容的精确度
          - context_recall: 检索内容的召回率
    
    应用场景:
        - 诊断质量评估
        - 系统性能监控
        - 持续改进依据
    """
    metrics = evaluate_qa_record(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    return {"metrics": metrics}


class ScoreRequest(BaseModel):
    question: str
    answer: str

class ScoreReview(BaseModel):
    critic: str
    score: float
    reason: str

class ScoreResponse(BaseModel):
    score: float
    reason: str
    reviews: List[ScoreReview] = []

@app.post("/api/score", response_model=ScoreResponse)
async def score_answer(req: ScoreRequest):
    """使用 ArkLLM 进行主观评分，返回分数、理由与多评审意见"""
    result = autogen_score(req.question, req.answer)
    return result


class RAGRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    top_k: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: list

 
# 健康检查
@app.get("/health")
async def health():
    """健康检查接口
    
    用于监控系统运行状态，检查服务是否正常响应。
    
    返回:
        - status: 服务状态，正常时为 "ok"
    
    用途:
        - 负载均衡器健康检查
        - 监控系统状态检测
        - 服务可用性验证
    """
    return {"status": "ok"} 