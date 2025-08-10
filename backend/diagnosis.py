"""backend.diagnosis
-------------------
终极优化版急诊智能诊断系统：
1. 所有文本输出均由大模型生成（无硬编码响应）
2. 医疗安全优先设计（高风险场景由LLM动态生成响应）
3. 完整的证据溯源与专业表述
4. 安全降级机制（最小化硬编码内容）
"""

from __future__ import annotations
import argparse
import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# 上述三行导入已移至 backend.tools.vector_search
from .llm import ArkLLM
from .config import ARK_API_KEY, EMBED_MODEL_DIR
from volcenginesdkarkruntime._exceptions import ArkError, ArkPermissionDeniedError
from .tools.vector_search import search_cases  # 迁移后路径

# 向量路径与缓存逻辑已迁移至 backend.tools.vector_search

# =============== 嵌入模型 ===============

# 删除旧版 _load_embed_model/_load_vector_db，完全使用 backend.tools.vector_search

# =============== 向量数据库 ===============

# 删除旧版 _load_embed_model/_load_vector_db，完全使用 backend.tools.vector_search


# 旧版向量检索接口已迁移至 backend.tools.vector_search
def _legacy_search_cases(query: str, top_k: int = 3):
    """(Deprecated) 兼容旧调用，内部直接调用新实现"""
    return search_cases(query, top_k)

# =============== 医疗安全核心模块 ===============

def _is_high_risk_query(query: str) -> Tuple[bool, str]:
    """检测高风险主诉（需立即干预）"""
    high_risk_patterns = [
        (r"(胸(痛|闷|梗)|心绞痛|心梗)", "心血管危象"),
        (r"(呼吸(困难|急促)|窒息|气促)", "呼吸衰竭"),
        (r"(大出(血|血))", "失血性休克"),
        (r"(意识(丧失|模糊)|晕厥|昏迷)", "意识障碍"),
        (r"(自(杀|残)|轻生)", "心理危机"),
        (r"(孕妇|妊娠|怀孕).*?疼痛", "妊娠期疼痛"),
        (r"(颈部|脊柱).*?损伤", "脊髓损伤风险")
    ]
    
    for pattern, risk_type in high_risk_patterns:
        if re.search(pattern, query):
            return True, risk_type
    
    return False, ""

def _build_high_risk_prompt(query: str, risk_type: str, context: List[Tuple[float, dict]]) -> str:
    """构建高风险场景的LLM提示词（所有响应均由LLM生成）"""
    retrieved = _format_retrieved_cases(context)
    
    return f"""
### 医疗AI系统指令（高风险场景）
您是一名急诊科主任医师，正在处理一个高风险急诊场景。系统检测到患者主诉可能涉及【{risk_type}】，需要您立即生成专业、结构化的紧急响应。

### 临床决策协议（必须严格遵守）
1. **【证据优先】** 仅基于下方"检索到的医学证据"作答，禁止虚构医学事实
2. **【紧急程度】** 必须明确标注紧急级别（1-5级）及理由
3. **【关键行动】** 列出必须立即执行的3项关键操作（按时间顺序）
4. **【禁忌核查】** 必须检查并明确指出治疗禁忌
5. **【证据标注】** 所有结论需标注证据等级（A/B/C）

### 患者主诉
{query}

### 检索到的医学证据（按证据等级+相似度排序）
{retrieved}

### 请严格按此结构输出（Markdown格式）
#### 紧急程度评估
- **分级：** [1-5级]
- **理由：** [基于证据的简要分析]

#### 立即行动指南
- **必须立即执行（0-5分钟）：**
    - [操作1]
    - [操作2]
    - [操作3]
- **5-15分钟内必须完成：**
    - [操作1]
    - [操作2]

#### 诊疗关键点
- **必须排除的危急疾病：**
    - [疾病1]：[排除指征，如"心电图正常+肌钙蛋白阴性"]
    - [疾病2]：[排除指征]
- **治疗禁忌：**
    - [明确禁忌]（依据：[evidence_level]级证据）
- **安全替代方案：**
    - [替代方案]（依据：[evidence_level]级证据）

#### 专业建议
[2-3句专业总结，强调关键风险和处理要点]

> 注意：若证据不足，必须写明"需进一步检查确认"，不可猜测
"""

def _build_safe_fallback_prompt(query: str, context: List[Tuple[float, dict]]) -> str:
    """构建安全回退模式的提示词（所有响应均由LLM生成）"""
    # 即使在回退模式，我们也尝试用LLM生成响应
    # 但提示词会说明当前处于系统降级状态
    
    retrieved = _format_retrieved_cases(context) if context else "无匹配医学证据"
    
    return f"""
### 医疗AI系统指令（安全回退模式）
您是一名急诊科主任医师。医疗AI系统当前处于降级状态（主服务不可用），但您仍需要基于有限的医学证据提供专业建议。

### 重要约束
1. **【仅基于证据】** 严格基于下方"检索到的医学证据"作答，禁止虚构
2. **【明确标注】** 必须注明"系统降级模式"和证据等级
3. **【安全优先】** 优先考虑最安全的处理方案
4. **【避免猜测】** 证据不足时明确说明需进一步检查

### 患者主诉
{query}

### 检索到的医学证据（可能不完整）
{retrieved}

### 请严格按此结构输出（Markdown格式）
#### 系统状态声明
- **当前状态：** 医疗AI系统降级模式
- **响应可靠性：** [高/中/低]（基于证据质量）
- **医生确认要求：** 必须由主治医师现场确认所有建议

#### 诊疗建议（基于现有证据）
- **紧急程度：** [1-5级]（依据：[简要说明]）
- **必须执行的操作：**
    - [操作1]（依据：[evidence_level]级证据）
    - [操作2]（依据：[evidence_level]级证据）
- **关键禁忌：**
    - [禁忌事项]（依据：[evidence_level]级证据）

#### 专业建议
[2-3句专业总结，强调系统状态限制和需医生确认的关键点]

> 注意：此响应由备用系统生成，所有医疗决策必须由专业医师确认
"""

def _format_retrieved_cases(context: List[Tuple[float, dict]]) -> str:
    """将检索结果转化为结构化医学知识（临床决策关键点）"""
    if not context:
        return "无匹配医学证据，请进行基础评估"
    
    results = []
    for rank, (score, case) in enumerate(context, 1):
        disease = case["full_data"]["disease"]
        
        # 提取关键临床决策点（按急诊优先级排序）
        key_features = disease["key_features"][:2]
        urgency_emoji = "🚨" if disease["urgency_level"] == "紧急" else "⚠️"
        evidence_emoji = "✅" if disease["evidence_level"] == "A" else "🟡"
        
        # 生成结构化摘要（符合临床思维流程）
        info = (
            f"{urgency_emoji} **【{disease['name']}】** (ICD: {disease['icd_code']})\n"
            f"   • 证据等级: {evidence_emoji} {disease['evidence_level']} | 相似度: {score:.0%}\n"
            f"   • **关键特征**: {', '.join(key_features)}\n"
        )
        
        # 添加检查建议（仅显示最高优先级）
        if disease["recommended_tests"]:
            test = disease["recommended_tests"][0]
            # 解析检查名称（兼容无"|"分隔的写法）
            if "test_name" in test and isinstance(test["test_name"], str):
                parts = test["test_name"].split("|", 1)
                test_display = parts[1].strip() if len(parts) > 1 else test["test_name"].strip()
            else:
                test_display = "检查项目未命名"
            info += f"   • **首选检查**: {test_display} (用于{test.get('purpose', '诊断')})\n"
        
        # 添加治疗核心点（强调禁忌）
        if disease["treatments"]:
            treatment = disease["treatments"][0]
            desc = treatment["description"]
            # 检测禁忌关键词
            has_contraindication = "禁忌" in desc or "禁用" in desc or "不推荐" in desc
            info += f"   • **{'⚠️ 首选治疗' if has_contraindication else '✅ 首选治疗'}**: {treatment['intervention']}\n"
            info += f"     - {desc[:100]}{'...' if len(desc) > 100 else ''}\n"
        
        results.append(f"{rank}. {info}")
    
    return "\n".join(results)

def _build_clinical_prompt(query: str, retrieved: str) -> str:
    """构建临床决策提示词（证据驱动+安全优先）"""
    return f"""
### 临床决策协议（必须严格遵守）
1. **【证据优先】** 仅基于下方"检索到的医学证据"作答，禁止虚构医学事实
2. **【危急鉴别】** 当症状匹配多个危急疾病时，必须列出Top2鉴别诊断及区分点
3. **【禁忌核查】** 必须检查治疗禁忌（如：非骨折性疼痛禁用阿片类）
4. **【证据标注】** 所有结论需标注证据等级（A/B/C）
5. **【高风险拦截】** 若主诉含胸痛/呼吸困难等，立即触发紧急协议

### 患者主诉
{query}

### 检索到的医学证据（按证据等级+相似度排序）
{retrieved}

### 请严格按此结构输出（Markdown格式）

#### 紧急程度分级
- **分级：** [1-5级] 
- **理由：** [引用证据中的urgency_level/key_features]
- **危急鉴别：** 
    - [疾病1]：[关键区分点，如"心电图ST段抬高"]
    - [疾病2]：[关键区分点]

#### 诊疗路径
- **必须排除的危急疾病：** 
    - [疾病]：[排除指征，如"心电图正常+肌钙蛋白阴性"]
- **首选检查：** 
    - [检查项目]（依据：[evidence_level]级证据）
    - [检查项目]（依据：[evidence_level]级证据）
- **治疗禁忌：** 
    - [明确禁忌]（依据：[evidence_level]级证据）
- **初步处理：** 
    - [基于A级证据的方案]
    - [安全替代方案（如禁忌存在）]

#### 医生行动建议
- **立即执行：** [2-3项关键操作]
- **30分钟内：** [必须完成的检查/处理]
- **需会诊科室：** [列出1-2个]

#### 专业总结
[2-3句专业总结，强调关键风险和处理要点]

> 注意：若证据不足，必须写明"需进一步检查确认"，不可猜测
"""

def _call_llm_with_fallback(prompt: str, fallback_prompt: str = None, history: List[dict] = None) -> str:
    """调用LLM，带智能回退机制"""
    try:
        llm = ArkLLM(api_key=ARK_API_KEY)
        return llm.invoke(prompt, history=history)
    except (ArkError, ArkPermissionDeniedError, Exception) as e:
        print(f"🔥 ArkLLM 主调用失败: {str(e)}")
        
        # 如果有备用提示词，尝试用备用提示词调用
        if fallback_prompt:
            try:
                print("🔄 尝试使用备用提示词重新调用...")
                llm = ArkLLM(api_key=ARK_API_KEY)
                return llm.invoke(fallback_prompt, history=history)
            except Exception as e2:
                print(f"🔥 ArkLLM 备用调用也失败: {str(e2)}")
        
        # 极端情况：连LLM都不可用，只能返回最简系统错误
        # 注意：这里只保留最基本的系统错误信息（无法避免的硬编码）
        return (
            "⚠️ **【系统严重错误】** 医疗AI服务完全不可用\n\n"
            "当前系统状态：\n"
            "- 主AI服务不可用\n"
            "- 备用服务不可用\n"
            "- 无法提供任何医疗建议\n\n"
            "**请立即联系系统管理员并进行人工评估**\n"
            "所有医疗决策必须由专业医师现场确认"
        )

def _call_llm(query: str, context: List[Tuple[float, dict]], history: List[dict] = None) -> str:
    """调用LLM生成诊疗建议（所有输出均由大模型生成）"""
    # 构建对话历史
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # [1] 高风险主诉处理 - 通过LLM生成响应
    is_high_risk, risk_type = _is_high_risk_query(query)
    if is_high_risk:
        print(f"🚨 检测到高风险主诉: {risk_type}")
        prompt = _build_high_risk_prompt(query, risk_type, context)
        return _call_llm_with_fallback(prompt, history=messages)
    
    # [2] 检查空查询
    if not query.strip():
        empty_prompt = (
            "### 医疗AI系统指令\n"
            "用户没有提供有效的患者主诉。请生成一个专业、友好的提示，指导用户如何正确描述症状。\n\n"
            "### 要求\n"
            "1. 用中文回答\n"
            "2. 保持专业但友好的语气\n"
            "3. 提供2-3个主诉描述的示例\n"
            "4. 不要指责用户，而是提供建设性指导"
        )
        return _call_llm_with_fallback(empty_prompt)
    
    # [3] 常规查询处理
    retrieved = _format_retrieved_cases(context)
    prompt = _build_clinical_prompt(query, retrieved)
    fallback_prompt = _build_safe_fallback_prompt(query, context)
    
    response = _call_llm_with_fallback(prompt, fallback_prompt, history=messages)
    
    # [4] 证据溯源（强制添加来源，但通过LLM生成）
    if context and "系统严重错误" not in response:
        source_prompt = (
            f"### 医疗AI系统指令\n"
            f"请为以下诊疗建议添加证据来源说明。只添加来源信息，不要修改原有内容。\n\n"
            f"### 原始诊疗建议\n"
            f"{response}\n\n"
            f"### 医学证据来源\n"
            f"依据：{context[0][1]['full_data']['disease']['source']} | 证据等级：{context[0][1]['full_data']['disease']['evidence_level']}\n\n"
            f"### 要求\n"
            f"1. 在原始建议末尾添加来源信息\n"
            f"2. 使用专业但不突兀的方式\n"
            f"3. 保持Markdown格式"
        )
        return _call_llm_with_fallback(source_prompt)
    
    return response

# =============== 对外接口 ===============

def get_diagnosis(query: str, history: List[dict] = None, top_k: int = 3) -> str:
    """获取结构化诊疗建议（所有输出均由大模型生成）"""
    print(f"🔍 正在分析主诉: '{query}'")
    
    # 检索医学知识
    cases = search_cases(query, top_k)
    print(f"📊 匹配到 {len(cases)} 条医学证据")

    # 生成诊疗建议（所有输出均由LLM生成）
    return _call_llm(query, cases, history)

# =============== CLI ===============

def _build_cli_error_prompt(error_msg: str) -> str:
    """构建CLI错误消息的LLM提示词"""
    return (
        f"### 医疗AI系统指令\n"
        f"用户在使用命令行界面时遇到了错误。请生成一个专业、清晰的错误说明和解决方案。\n\n"
        f"### 错误信息\n"
        f"{error_msg}\n\n"
        f"### 要求\n"
        f"1. 用中文回答\n"
        f"2. 解释错误原因（技术层面和用户操作层面）\n"
        f"3. 提供2-3个具体解决方案\n"
        f"4. 保持专业但友好的语气\n"
        f"5. 不要包含技术术语，面向医疗专业人员解释"
    )

def _cli() -> None:
    """命令行接口（所有输出均由大模型生成）"""
    try:
        parser = argparse.ArgumentParser(
            description="急诊智能诊断系统 - 证据驱动的临床决策支持",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "query",
            help="患者主诉/症状描述（例: '60岁男性突发胸痛30分钟'）",
        )
        parser.add_argument("--top_k", type=int, default=3, help="返回医学证据数量 (默认: 3)")
        parser.add_argument("--debug", action="store_true", help="显示详细检索结果")
        args = parser.parse_args()

        # 执行诊断
        print("\n🔄 正在进行临床决策分析...")
        answer = get_diagnosis(args.query, top_k=args.top_k)

        # 通过LLM生成标题（确保一致性）
        header_prompt = (
            "### 医疗AI系统指令\n"
            "请为以下急诊临床决策建议生成一个专业、简洁的标题。\n\n"
            "### 诊疗建议内容\n"
            f"{answer[:500]}...\n\n"  # 只提供部分内容供参考
            "### 要求\n"
            "1. 标题应反映患者主诉和紧急程度\n"
            "2. 包含紧急程度标识（如'【紧急】'）\n"
            "3. 长度不超过20个字\n"
            "4. 使用专业但易懂的医学术语"
        )
        header = _call_llm_with_fallback(header_prompt)

        # 显示结果
        print("\n" + "=" * 50)
        print(f"{header.strip()}")
        print("=" * 50)
        print(answer)

        # 调试模式显示检索结果
        if args.debug:
            debug_prompt = (
                "### 医疗AI系统指令\n"
                "请解释以下医学证据对当前诊断的意义。面向急诊科医生提供专业解读。\n\n"
                "### 检索到的医学证据\n"
                f"{_format_retrieved_cases(search_cases(args.query, 10))}\n\n"
                "### 患者主诉\n"
                f"{args.query}\n\n"
                "### 要求\n"
                "1. 分析每条证据的相关性和可靠性\n"
                "2. 指出哪些证据对当前诊断最关键\n"
                "3. 用专业但简洁的语言\n"
                "4. 保持Markdown格式"
            )
            debug_info = _call_llm_with_fallback(debug_prompt)

            print("\n🔍 检索证据专业解读:")
            print(debug_info)

    except Exception as e:
        # 所有错误消息也通过LLM生成
        error_msg = f"命令行界面发生错误: {str(e)}"
        print("\n❌ 发生系统错误，正在生成专业错误说明...")
        error_response = _call_llm_with_fallback(_build_cli_error_prompt(error_msg))
        print("\n" + "=" * 50)
        print("❌ 系统错误说明")
        print("=" * 50)
        print(error_response)

if __name__ == "__main__":
    _cli() 