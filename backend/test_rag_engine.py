"""
RAG 引擎测试 — 测试文档加载、分块、索引构建、检索、删除
运行: pytest backend/test_rag_engine.py -v
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from rag_engine import RAGEngine


@pytest.fixture
def engine():
    """使用临时目录创建引擎实例"""
    tmp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(tmp_dir)

    engine = RAGEngine(
        api_key=os.getenv("API_KEY", "test-key"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )

    yield engine

    os.chdir(original_cwd)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sample_txt():
    """创建一个测试 TXT 文件"""
    content = """RAG（Retrieval-Augmented Generation）是一种结合检索和生成的 AI 技术。

它首先从知识库中检索相关文档片段，然后将这些片段作为上下文提供给大语言模型，
让模型基于检索到的信息生成回答。这种方式可以有效减少幻觉，提高回答的准确性。

RAG 系统通常包含三个核心组件：
1. 文档加载器：负责读取各种格式的文档
2. 向量化引擎：将文本转换为向量表示
3. 检索引擎：根据用户问题找到最相关的文档片段"""

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestDocumentLoading:
    """文档加载测试"""

    def test_load_txt(self, engine, sample_txt):
        docs = engine.load_document(sample_txt)
        assert len(docs) > 0
        assert "RAG" in docs[0].page_content
        assert docs[0].metadata["source"] == os.path.basename(sample_txt)

    def test_load_unsupported_format(self, engine):
        with pytest.raises(ValueError, match="不支持的文件格式"):
            engine.load_document("/tmp/test.xyz")


class TestChunking:
    """分块测试"""

    def test_split_documents(self, engine, sample_txt):
        docs = engine.load_document(sample_txt)
        chunks = engine.text_splitter.split_documents(docs)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.page_content) > 0


class TestIndexAndRetrieve:
    """索引与检索测试"""

    def test_build_and_retrieve(self, engine, sample_txt):
        chunk_count = engine.build_index([sample_txt])
        assert chunk_count > 0

        docs = engine.retrieve("什么是RAG", top_k=2)
        assert len(docs) > 0
        assert len(docs) <= 2
        assert docs[0].metadata.get("score") is not None


class TestDocumentManagement:
    """文档管理测试"""

    def test_list_documents(self, engine, sample_txt):
        engine.build_index([sample_txt])
        docs = engine.list_documents()
        assert len(docs) > 0
        assert any(os.path.basename(sample_txt) == d["filename"] for d in docs)

    def test_delete_document(self, engine, sample_txt):
        engine.build_index([sample_txt])
        fname = os.path.basename(sample_txt)

        docs_before = engine.list_documents()
        assert any(d["filename"] == fname for d in docs_before)

        deleted = engine.delete_document(fname)
        assert deleted > 0

        docs_after = engine.list_documents()
        assert not any(d["filename"] == fname for d in docs_after)


class TestEmptyIndex:
    """空索引测试"""

    def test_list_empty(self, engine):
        assert engine.list_documents() == []

    def test_retrieve_empty_raises(self, engine):
        with pytest.raises(RuntimeError, match="向量库未初始化"):
            engine.retrieve("test query")
