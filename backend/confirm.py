from __future__ import annotations

"""backend.confirm
------------------
提供检验结果分析与确诊功能。
所有业务逻辑委托给 ArkLLM 生成，Python 端仅做提示词构建与结果解析，
若调用失败则返回安全降级的占位内容，确保 API 可用。
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .llm import ArkLLM
from .config import ARK_API_KEY
from volcenginesdkarkruntime._exceptions import ArkError, ArkPermissionDeniedError


@dataclass
class LabResult:
    """实验室/检查结果条目"""

    test_name: str
    result_value: str
    unit: str | None = None
    is_abnormal: bool = False


@dataclass
class AnalysisResult:
    """智能体分析返回结构"""

    updated_diagnosis: str
    confidence_change: str
    next_steps: List[str]
    urgent_actions: List[str]
    new_evidence: List[LabResult] = field(default_factory=list)


def _safe_analysis_fallback() -> AnalysisResult:
    """当LLM调用失败时返回的占位内容，保证接口稳定"""
    return AnalysisResult(
        updated_diagnosis="无法生成更新的诊断，请人工评估",
        confidence_change="0%",
        next_steps=["复查网络连接", "稍后重试"],
        urgent_actions=[],
        new_evidence=[],
    )


def _extract_json_from_response(raw_response: str) -> Dict[str, Any]:
    """从LLM响应中提取JSON，处理各种格式"""
    # 清理响应
    cleaned = raw_response.strip()
    
    # 尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 ```json ... ``` 格式
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取 ``` ... ``` 格式（无json标识）
    code_match = re.search(r'```\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取 { ... } 格式
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法从响应中提取有效JSON: {raw_response[:200]}...")


def analyze_medical_case(history: List[Dict[str, str]], test_results: str) -> AnalysisResult:
    """结合对话历史与检查结果，对病情进行再评估。

    参数
    ------
    history: 聊天历史，列表，每条包含 role / content
    test_results: 最新检查结果（自由文本）
    """
    # 输入验证
    if not test_results.strip():
        print("⚠️ 检查结果为空，返回降级响应")
        return _safe_analysis_fallback()
    
    # 构建对话历史文本
    dialogue = "\n".join(
        [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in history]
    )

    prompt = f"""
### 医疗AI系统指令
你是一名急诊科主任医师，请根据以下对话历史与最新检查结果，更新诊断并给出后续行动建议。

### 对话历史
{dialogue if dialogue else '（无）'}

### 最新检查结果
{test_results}

### 输出要求（JSON 串）
{{
  "updated_diagnosis": "更新后的诊断结论（Markdown 可用）",
  "confidence_change": "置信度变化，如 +12% 或 -5%",
  "next_steps": ["建议下一步操作1", "建议下一步操作2"],
  "urgent_actions": ["必须立即执行的动作1", "必须立即执行的动作2"],
  "new_evidence": [
    {{
      "test_name": "检查项目名称",
      "result_value": "检查结果值",
      "unit": "单位（可选）",
      "is_abnormal": true
    }}
  ]
}}

请严格按照上述JSON格式输出，不要添加任何额外解释。
"""

    try:
        print(f"🔍 调用LLM分析检查结果，历史长度: {len(history)}")
        llm = ArkLLM(api_key=ARK_API_KEY)
        raw = llm.invoke(prompt)
        print(f"📝 LLM原始响应长度: {len(raw)}")
        
        # 提取JSON
        data = _extract_json_from_response(raw)
        print(f"✅ JSON解析成功，包含字段: {list(data.keys())}")
        
        # 构造数据类
        ev_list = []
        for e in data.get("new_evidence", []):
            try:
                ev_list.append(LabResult(
                    test_name=str(e.get("test_name", "")),
                    result_value=str(e.get("result_value", "")),
                    unit=e.get("unit"),
                    is_abnormal=bool(e.get("is_abnormal", False)),
                ))
            except Exception as ev_err:
                print(f"⚠️ 解析证据条目失败: {ev_err}")
                continue
        
        result = AnalysisResult(
            updated_diagnosis=str(data.get("updated_diagnosis", "")),
            confidence_change=str(data.get("confidence_change", "")),
            next_steps=list(data.get("next_steps", [])),
            urgent_actions=list(data.get("urgent_actions", [])),
            new_evidence=ev_list,
        )
        print(f"✅ 分析完成，置信度变化: {result.confidence_change}")
        return result
        
    except Exception as e:
        print(f"🔥 分析过程出错: {type(e).__name__}: {str(e)}")
        return _safe_analysis_fallback()


def confirm_disease_case(diagnosis_md: str, test_results: str) -> str:
    """结合首选检查结果给出最终确诊意见。

    返回完整 Markdown 文本，由大模型生成。
    """
    # 输入验证
    if not test_results.strip():
        return "⚠️ **输入错误**：检查结果为空，无法进行确诊分析。"
    
    if not diagnosis_md.strip():
        return "⚠️ **输入错误**：初步诊断为空，无法进行确诊分析。"

    prompt = f"""
### 医疗AI系统指令
你是一名急诊科主任医师。请根据以下"初步诊疗建议"和"首选检查结果"，给出最终确诊结论、治疗方案以及注意事项。

### 初步诊疗建议
{diagnosis_md}

### 首选检查结果
{test_results}

### 输出要求
1. 使用 Markdown 格式
2. 首段给出明确的疾病名称和严重程度标识
3. 分段列出：确诊依据、治疗方案、禁忌事项、预后评估
4. 若证据不足，必须明确指出需进一步检查
5. 保持专业、准确的医学表述
"""
    try:
        print(f"🔍 调用LLM进行确诊分析")
        llm = ArkLLM(api_key=ARK_API_KEY)
        result = llm.invoke(prompt)
        print(f"✅ 确诊分析完成，响应长度: {len(result)}")
        return result
    except Exception as e:
        print(f"🔥 确诊分析失败: {type(e).__name__}: {str(e)}")
        return (
            "⚠️ **系统降级**：无法生成确诊报告，请专业医师根据实际检查结果进行评估。"
        ) 