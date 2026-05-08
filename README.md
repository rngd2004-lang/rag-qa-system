# RAG 知识库问答系统

基于检索增强生成（RAG）的智能问答系统。上传 PDF/Word/TXT/Markdown 文档，基于文档内容问答，回答带引用来源。

## 功能

- 多格式文档：PDF、DOCX、TXT、Markdown
- 向量语义检索，自动定位最相关片段
- 流式输出，逐字显示
- 引用溯源，标注信息来源
- Gradio 可视化界面

## 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | DeepSeek / OpenAI |
| RAG | LangChain |
| 向量库 | Chroma |
| 后端 | FastAPI |
| 前端 | Gradio |
| 部署 | Docker Compose |

## 快速开始

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key (https://platform.deepseek.com/)

# 2. 安装依赖
pip install -r backend/requirements.txt
pip install gradio requests

# 3. 启动后端 (终端1)
cd backend && python main.py
# → http://localhost:8000

# 4. 启动前端 (终端2)
cd frontend && python app.py
# → http://localhost:7860
```

## Docker 一键启动

```bash
docker compose up -d
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /upload | 上传文档 |
| POST | /chat | 对话 (SSE 流式) |

## 项目结构

```
rag-qa-system/
├── backend/
│   ├── main.py          # FastAPI
│   ├── rag_engine.py    # RAG 引擎
│   └── requirements.txt
├── frontend/
│   └── app.py           # Gradio UI
├── data/                # 上传文件 + 向量库
├── docker-compose.yml
└── README.md
```
