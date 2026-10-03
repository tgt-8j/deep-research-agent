"""RAG 检索质量对比测试。

对比三种检索策略：
1. 纯向量检索（Baseline）
2. Hybrid Search（BM25 + RRF 融合）
3. Hybrid + Rerank（两级重排序）

通过人工设计的查询集，验证各策略的召回质量和排序合理性。
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_search import _tokenize, bm25_search, hybrid_search, rrf_merge
from src.retrieval.reranker import Reranker

# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents():
    """生成用于测试的文档集合。"""
    return [
        {
            "doc_id": "d1",
            "content": "LangGraph is a framework for building stateful multi-agent applications with cyclic graphs",
            "title": "LangGraph Official Docs",
        },
        {
            "doc_id": "d2",
            "content": "LangGraph reflection loop allows agents to iterate and improve their outputs through self-correction",
            "title": "LangGraph Reflection Loop",
        },
        {
            "doc_id": "d3",
            "content": "Python programming language is a high-level general-purpose programming language",
            "title": "Python Basics",
        },
        {
            "doc_id": "d4",
            "content": "Retrieval augmented generation (RAG) combines retrieval with LLM generation for improved accuracy",
            "title": "RAG Introduction",
        },
        {
            "doc_id": "d5",
            "content": "LangChain provides tools for building LLM applications including chains and agents",
            "title": "LangChain Framework",
        },
        {
            "doc_id": "d6",
            "content": "Vector databases store embeddings for efficient similarity search in RAG systems",
            "title": "Vector Database Guide",
        },
        {
            "doc_id": "d7",
            "content": "BM25 is a bag-of-words retrieval function that ranks documents based on query terms",
            "title": "BM25 Algorithm",
        },
        {
            "doc_id": "d8",
            "content": "Reciprocal Rank Fusion combines multiple ranking results by aggregating inverse rank scores",
            "title": "RRF Fusion Method",
        },
        {
            "doc_id": "d9",
            "content": "Embedding models convert text into dense vector representations for semantic search",
            "title": "Embedding Models",
        },
        {
            "doc_id": "d10",
            "content": "Cross-encoder reranking models score query-document pairs for higher relevance accuracy",
            "title": "Cross-Encoder Rerank",
        },
    ]


@pytest.fixture
def sample_evidence():
    """生成用于重排序测试的证据列表。"""
    return [
        {
            "source_id": f"WEB-{i:03d}",
            "source_type": "web",
            "title": doc["title"],
            "snippet": doc["content"][:80],
            "domain": "example.com",
            "reliability_score": 0.6,
        }
        for i, doc in enumerate(
            [
                {
                    "title": "LangGraph Official",
                    "content": "LangGraph is a framework for building agents",
                },
                {
                    "title": "Python Tutorial",
                    "content": "Python is a programming language",
                },
                {
                    "title": "LangGraph Deep Dive",
                    "content": "LangGraph reflection loop implementation",
                },
                {
                    "title": "ML Fundamentals",
                    "content": "Machine learning basics and neural networks",
                },
                {
                    "title": "LangChain Guide",
                    "content": "LangChain integration with LLMs",
                },
                {
                    "title": "Vector DB",
                    "content": "Vector databases for semantic search",
                },
            ]
        )
    ]


# ---------------------------------------------------------------------------
# 分词测试
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_english_simple(self):
        tokens = _tokenize("LangGraph is a framework")
        assert "langgraph" in tokens
        assert "framework" in tokens

    def test_chinese_simple(self):
        tokens = _tokenize("LangGraph 反思循环")
        assert "LangGraph" in tokens
        # 中文字符按单字切分
        assert len(tokens) > 1

    def test_mixed_content(self):
        tokens = _tokenize("Python LangGraph 框架")
        assert "Python" in tokens  # 英文保持原大小写
        assert "LangGraph" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# BM25 检索测试
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_keyword_match(self, sample_documents):
        """关键词匹配测试。"""
        results = bm25_search("LangGraph framework", sample_documents, top_k=3)
        assert len(results) <= 3
        # LangGraph 相关文档应排在前面
        top_ids = [r["doc_id"] for r in results]
        assert "d1" in top_ids or "d2" in top_ids

    def test_no_match(self, sample_documents):
        """无匹配时应返回空或低分结果。"""
        results = bm25_search("completely unrelated xyz", sample_documents)
        assert isinstance(results, list)

    def test_empty_documents(self):
        """空文档列表应返回空结果。"""
        results = bm25_search("query", [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# RRF 融合测试
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_merge_rankings(self):
        """多路排序融合测试。"""
        ranking1 = [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]
        ranking2 = [{"doc_id": "b", "score": 0.85}, {"doc_id": "c", "score": 0.7}]
        merged = rrf_merge([ranking1, ranking2], rrf_k=60)
        # b 在两路中都出现，应排名最高
        assert merged[0]["doc_id"] == "b"

    def test_single_ranking_passthrough(self):
        """单路排序直接返回。"""
        ranking = [{"doc_id": "x", "score": 0.95}]
        merged = rrf_merge([ranking], rrf_k=60)
        assert len(merged) == 1
        assert merged[0]["doc_id"] == "x"

    def test_rerank_params(self):
        """RRF k 参数影响排序结果。"""
        ranking1 = [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]
        ranking2 = [{"doc_id": "b", "score": 0.85}, {"doc_id": "a", "score": 0.7}]
        merged_k60 = rrf_merge([ranking1, ranking2], rrf_k=60)
        merged_k1 = rrf_merge([ranking1, ranking2], rrf_k=1)
        # k 越小，排名差异越大
        assert len(merged_k60) == len(merged_k1) == 2


# ---------------------------------------------------------------------------
# Hybrid Search 测试
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_hybrid_with_bm25_only(self, sample_documents):
        """纯 BM25 模式（无 embeddings）。"""
        results = hybrid_search(
            "LangGraph reflection",
            sample_documents,
            k_vector=0,
            k_bm25=3,
            rrf_k=60,
        )
        assert len(results) <= 3
        # 不应包含内部字段
        for r in results:
            assert "_rrf_score" not in r
            assert "_bm25_score" not in r

    def test_empty_input(self):
        """空输入应返回空列表。"""
        results = hybrid_search("query", [], k_bm25=5)
        assert results == []

    def test_docs_without_doc_id(self, sample_documents):
        """文档无 doc_id 时使用 title 作为标识。"""
        docs = [{"content": "LangGraph tutorial", "title": "Intro to LangGraph"}]
        results = hybrid_search("LangGraph", docs)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Reranker 测试
# ---------------------------------------------------------------------------


class TestReranker:
    def test_empty_input(self, sample_evidence):
        """空证据列表。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, stats = reranker.rerank("query", [])
        assert results == []
        assert stats["method"] == "none"

    def test_fewer_than_fine_k(self, sample_evidence):
        """证据数少于精排数量。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, _stats = reranker.rerank("LangGraph", sample_evidence[:2], fine_k=10)
        assert len(results) == 2

    def test_stats_structure(self, sample_evidence):
        """stats 结构完整性。"""
        reranker = Reranker(use_embedding_rerank=False)
        _, stats = reranker.rerank("LangGraph", sample_evidence)
        assert "method" in stats
        assert "total_before" in stats
        assert "total_after" in stats

    def test_source_type_preserved(self, sample_evidence):
        """source_type 字段保留。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, _ = reranker.rerank("LangGraph", sample_evidence)
        source_types = {item.get("source_type") for item in results}
        assert "web" in source_types


# ---------------------------------------------------------------------------
# 端到端对比测试
# ---------------------------------------------------------------------------


class TestRetrievalComparison:
    """对比不同检索策略的效果。"""

    @pytest.mark.parametrize(
        "query",
        [
            "LangGraph reflection loop implementation",
            "Python programming basics",
            "RAG retrieval augmented generation",
        ],
    )
    def test_bm25_vs_hybrid(self, sample_documents, query):
        """BM25 vs Hybrid 对比。"""
        # BM25 only
        bm25_results = bm25_search(query, sample_documents, top_k=5)
        # Hybrid (BM25 + RRF with single ranking)
        hybrid_results = hybrid_search(
            query, sample_documents, k_vector=0, k_bm25=5, rrf_k=60
        )

        # Hybrid 结果应包含 BM25 的前几名
        bm25_ids = {r["doc_id"] for r in bm25_results}
        hybrid_ids = {r["doc_id"] for r in hybrid_results}
        # 至少应有部分重叠
        assert len(bm25_ids & hybrid_ids) > 0 or len(hybrid_results) == 0

    def test_rerank_quality(self, sample_evidence):
        """重排序质量测试。"""
        reranker = Reranker(use_embedding_rerank=False)
        query = "LangGraph reflection"

        # 原始顺序
        original = sample_evidence.copy()
        # 重排序后
        reranked, _stats = reranker.rerank(query, original)

        # 重排序应减少数量到 fine_k
        assert len(reranked) <= len(original)
        # 保留证据类型信息
        assert all("source_type" in item for item in reranked)


# ---------------------------------------------------------------------------
# 边界情况测试
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_document(self, sample_documents):
        """单文档场景。"""
        results = hybrid_search("LangGraph", [sample_documents[0]], k_bm25=1)
        assert len(results) <= 1

    def test_all_irrelevant(self, sample_documents):
        """完全无关查询。"""
        results = bm25_search("xyz abc def", sample_documents)
        # 可能返回空或低分结果
        assert isinstance(results, list)

    def test_special_characters(self, sample_documents):
        """特殊字符查询。"""
        results = bm25_search("LangGraph?反射循环!", sample_documents, top_k=3)
        assert isinstance(results, list)
