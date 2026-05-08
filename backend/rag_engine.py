"""
RAG 核心引擎 — 文档加载、分块、向量化、检索、生成回答
"""

import os
import logging
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger("rag_engine")

# ── 可配置参数 ──────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ── RAG 提示词 ──────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""你是一个基于知识库的智能问答助手。
请严格根据以下检索到的文档内容回答问题。如果文档中没有相关信息，请如实说"文档中未找到相关信息"，不要编造。

【检索到的文档内容】
{context}

【用户问题】
{question}

【回答要求】
1. 基于文档内容回答，不要添加文档中没有的信息
2. 回答末尾注明信息来源（文档名 + 相关片段摘录）
3. 如果文档内容不足以回答问题，直接说明

回答：""")


class RAGEngine:
    """RAG 引擎：管理知识库索引、检索与生成"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.vectorstore: Optional[Chroma] = None
        self._init_components()

    def _init_components(self):
        """初始化 LLM / Embedding / 文本分割器"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.3,
            streaming=True,
        )

        # 使用本地 HuggingFace 嵌入模型（免费、中文优化、无需 API）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=True,
        )

    # ── 文档加载 ───────────────────────────────────────

    def load_document(self, file_path: str) -> List[Document]:
        """根据扩展名选择加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        loader_map = {
            ".pdf":  PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt":  TextLoader,
            ".md":   UnstructuredMarkdownLoader,
        }
        loader_cls = loader_map.get(ext)
        if loader_cls is None:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {list(loader_map.keys())}")

        loader = loader_cls(file_path)
        docs = loader.load()
        fname = os.path.basename(file_path)
        for d in docs:
            d.metadata["source"] = fname
        logger.info(f"已加载: {fname}, {len(docs)} 页/段")
        return docs

    # ── 索引构建 ───────────────────────────────────────

    def build_index(self, file_paths: List[str]) -> int:
        """加载多个文档 → 分块 → 向量化 → 存入 Chroma"""
        all_docs = []
        for fp in file_paths:
            all_docs.extend(self.load_document(fp))

        if not all_docs:
            return 0

        chunks = self.text_splitter.split_documents(all_docs)
        logger.info(f"分块完成: {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        return len(chunks)

    def load_index(self) -> int:
        """从磁盘加载已有向量库"""
        if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
            )
            cnt = self.vectorstore._collection.count()
            logger.info(f"已加载向量库: {cnt} 条")
            return cnt
        return 0

    # ── 检索 ──────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        """向量相似度检索"""
        if self.vectorstore is None:
            raise RuntimeError("向量库未初始化，请先上传文档")
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        docs = []
        for doc, score in docs_with_scores:
            doc.metadata["score"] = f"{score:.4f}"
            docs.append(doc)
        return docs

    # ── 生成回答 ───────────────────────────────────────

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "未知")
            score = doc.metadata.get("score", "N/A")
            parts.append(f"[片段{i}] 来源: {src} | 相似度: {score}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, docs: List[Document]) -> str:
        seen = {}
        for doc in docs:
            src = doc.metadata.get("source", "未知")
            if src not in seen:
                seen[src] = doc.page_content[:100]
        return "\n".join(
            f"- {src}: \"{excerpt}...\"" for src, excerpt in seen.items()
        )

    async def ask(self, question: str, top_k: int = TOP_K):
        """检索 → 拼接上下文 → LLM 流式生成"""
        docs = self.retrieve(question, top_k=top_k)
        context = self._format_context(docs)
        sources = self._format_sources(docs)

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        full_answer = ""
        async for chunk in chain.astream(question):
            full_answer += chunk
            yield {"type": "token", "content": chunk}

        yield {"type": "sources", "content": sources, "docs": docs}
        yield {"type": "done", "content": full_answer}
