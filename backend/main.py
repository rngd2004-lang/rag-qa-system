"""
FastAPI 后端 — RAG 知识库问答系统 API
"""

import os
import sys
import logging
import shutil
import uuid

# 确保工作目录是项目根目录（无论从哪里运行 python backend/main.py）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "backend"))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGEngine

# ── 日志 ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")

# ── FastAPI 应用 ───────────────────────────────────────
app = FastAPI(title="RAG 知识库问答系统", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 配置 ───────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL", "deepseek-chat")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 初始化 RAG 引擎 ───────────────────────────────────
engine = RAGEngine(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
)
indexed_count = engine.load_index()
logger.info(f"向量库已有 {indexed_count} 条记录")


# ── 请求模型 ──────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict] | None = None


# ── API ────────────────────────────────────────────────

@app.get("/health")
def health():
    cnt = engine.vectorstore._collection.count() if engine.vectorstore else 0
    return {"status": "ok", "indexed_chunks": cnt}


@app.get("/documents")
def list_documents():
    """列出所有已索引的文档"""
    docs = engine.list_documents()
    return {"status": "ok", "documents": docs, "total": len(docs)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """删除指定文档及其所有索引"""
    try:
        deleted = engine.delete_document(filename)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if deleted == 0:
        raise HTTPException(404, f"文档不存在或无索引: {filename}")
    return {"status": "ok", "filename": filename, "chunks_deleted": deleted}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    if not files or (len(files) == 1 and files[0].filename == ""):
        raise HTTPException(400, "请选择至少一个文件")

    existing_docs = {d["filename"] for d in engine.list_documents()}
    skipped = []
    saved_paths = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".pdf", ".docx", ".txt", ".md"):
            raise HTTPException(400, f"不支持的文件格式: {f.filename} ({ext})")

        base_name = f.filename
        if base_name in existing_docs:
            skipped.append(base_name)
            continue

        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{base_name}")
        with open(save_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(save_path)
        logger.info(f"已保存: {save_path}")

    chunk_count = 0
    if saved_paths:
        chunk_count = engine.build_index(saved_paths)

    result = {
        "status": "ok",
        "uploaded": len(saved_paths),
        "chunks_indexed": chunk_count,
        "files": [os.path.basename(p) for p in saved_paths],
    }
    if skipped:
        result["skipped"] = skipped
        result["message"] = f"{len(skipped)} 个文件已存在，已跳过"
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    async def event_stream():
        try:
            async for event in engine.ask(req.question, req.top_k, history=req.history):
                if event["type"] == "token":
                    yield f"data: {event['content']}\n\n"
                elif event["type"] == "sources":
                    yield f"data: [SOURCES]\n{event['content']}\n\n"
                elif event["type"] == "done":
                    yield "data: [DONE]\n\n"
        except RuntimeError as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
