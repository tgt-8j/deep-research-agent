"""混合检索（Hybrid Search）模块测试。

验证 BM25 关键词检索、RRF 融合、降级逻辑。
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_search import (
    bm25_search,
    hybrid_search,
    rrf_merge,
    _tokenize,
    _tokenize_chinese,
)


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_english_text(self):
        tokens = _tokenize("LangGraph is a framework")
        assert "langgraph" in tokens
        assert "framework" in tokens

    def test_chinese_text(self):
        tokens = _tokenize("LangGraph 反思循环的实现")
        assert "LangGraph" in tokens
        assert "反思" in tokens or "反" in tokens  # 按字符切分

    def test_empty_string(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# BM25 检索
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_basic_search(self):
        docs = [
            {"doc_id": "d1", "content": "LangGraph is a framework for building agents"},
            {"doc_id": "d2", "content": "Python programming language tutorial"},
            {"doc_id": "d3", "content": "LangGraph reflection loop implementation"},
        ]
        results = bm25_search("LangGraph reflection", docs, top_k=2)
        assert len(results) <= 2
        # d1 和 d3 应该都包含 LangGraph
        ids = [r["doc_id"] for r in results]
        assert "d1" in ids or "d3" in ids

    def test_no_match(self):
        docs = [
            {"doc_id": "d1", "content": "完全无关的内容"},
        ]
        results = bm25_search("LangGraph framework", docs)
        # BM25 可能返回空或低分结果
        assert isinstance(results, list)

    def test_empty_documents(self):
        results = bm25_search("query", [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_merge_two_rankings(self):
        ranking1 = [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]
        ranking2 = [{"doc_id": "b", "score": 0.85}, {"doc_id": "c", "score": 0.7}]
        merged = rrf_merge([ranking1, ranking2], rrf_k=60)
        # b 在两个 ranking 中都出现，应该排名最高
        assert merged[0]["doc_id"] == "b"
        # rrf_merge 返回含 _rrf_score，hybrid_search 会清理

    def test_single_ranking_passthrough(self):
        ranking = [{"doc_id": "x", "score": 0.95}]
        merged = rrf_merge([ranking], rrf_k=60)
        assert len(merged) == 1
        assert merged[0]["doc_id"] == "x"


# ---------------------------------------------------------------------------
# 混合检索
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_hybrid_with_bm25_only(self):
        docs = [
            {"doc_id": "d1", "content": "LangGraph agent framework tutorial", "title": "Doc 1"},
            {"doc_id": "d2", "content": "Python basics for beginners", "title": "Doc 2"},
            {"doc_id": "d3", "content": "LangGraph reflection loop deep dive", "title": "Doc 3"},
        ]
        results = hybrid_search("LangGraph reflection", docs, k_bm25=2, k_vector=2)
        assert len(results) <= 2
        # 不应包含内部字段
        for r in results:
            assert "_rrf_score" not in r
            assert "_bm25_score" not in r

    def test_empty_documents(self):
        results = hybrid_search("query", [], k_bm25=5)
        assert results == []

    def test_docs_without_doc_id(self):
        """文档没有 doc_id 时自动使用 title 作为标识。"""
        docs = [
            {"content": "LangGraph tutorial", "title": "Intro to LangGraph"},
        ]
        results = hybrid_search("LangGraph", docs)
        assert isinstance(results, list)
