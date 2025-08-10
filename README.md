# Doctor - 智能医疗诊断助手

基于大语言模型的智能医疗诊断系统，支持症状分析、检查建议和诊断评估。

## 🌟 特性

- 💡 **智能诊断**: 基于症状描述提供初步诊断建议
- 🔍 **证据检索**: 从医学指南中检索相关证据支持
- 📊 **质量评估**: 使用 Ragas 评估诊断建议的质量
- 🏥 **医疗安全**: 内置高风险场景识别与处理机制
- 🔄 **实时分析**: 支持检查结果实时分析与诊断更新

## 📦 安装

1. 创建并激活虚拟环境（推荐）
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境变量
   ```bash
   # Windows PowerShell
   $env:ARK_API_KEY = "你的-火山方舟-API-密钥"
   
   # Linux/macOS
   export ARK_API_KEY="你的-火山方舟-API-密钥"
   ```

## 🚀 快速开始

1. 启动后端服务cd backend
   ```bash
   uvicorn backend.server:app --reload --port 3001
   ```

2. 访问 API 文档
   ```
   http://localhost:8000/docs
   ```

## 📝 API 示例

### 1. 诊断建议

```bash
curl -X POST http://localhost:8000/api/diagnosis \
     -H "Content-Type: application/json" \
     -d '{
       "query": "心悸胸闷伴头晕",
       "history": [],
       "top_k": 5
     }'
```

### 2. 检查结果分析

```bash
curl -X POST http://localhost:8000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "history": [...],
       "test_results": "心电图显示：窦性心律，心率76次/分..."
     }'
```

### 3. 诊断质量评估

```bash
curl -X POST http://localhost:8000/api/evaluate \
     -H "Content-Type: application/json" \
     -d '{
       "question": "房颤患者需要做哪些检查？",
       "answer": "建议 12 导联心电图、BNP/NT-proBNP 等。",
       "contexts": [
         "…心电图显示窦性 P 波消失…",
         "…BNP 或 NT-proBNP 显著增高…"
       ],
       "ground_truth": "12 导联心电图、心肌酶、BNP/NT-proBNP 等"
     }'
```

## 🔧 项目结构

```
Docter/
├── backend/                # 后端代码
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── diagnosis.py       # 诊断核心逻辑
│   ├── llm.py            # 大模型封装
│   ├── server.py         # FastAPI 服务器
│   ├── ragas_evaluator.py # Ragas 评估工具
│   └── tools/            # 工具函数
├── frontend/             # 前端代码（可选） npm run dev
├── vector_db/           # 向量数据库 
└── requirements.txt     # Python 依赖 
```

## 📊 评估指标

使用 Ragas 评估诊断质量：

- **Faithfulness**: 答案对检索内容的忠实度
- **Answer Relevancy**: 答案与问题的相关性
- **Context Precision**: 检索内容的精确度
- **Context Recall**: 检索内容的召回率





首先创建环境变量.
然后cd 到这个Docter,使用 pip install -r requirements.txt


启动后端：
 
uvicorn backend.server:app --reload --port 8000

启动前端：
cd frontend 
npm run dev

文件中的test.py是用于评测RAG的示例代码

ps:本项目中暂时删除了向量化模型
