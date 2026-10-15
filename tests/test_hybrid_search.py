"""混合检索（Hybrid Search）模块测试。

验证 BM25 关键词检索、RRF 融合、降级逻辑。
"""

from __future__ import annotations

from src.retrieval.hybrid_search import (
    _tokenize,
    bm25_search,
    hybrid_search,
    rrf_merge,
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
            {
                "doc_id": "d1",
                "content": "LangGraph agent framework tutorial",
                "title": "Doc 1",
            },
            {
                "doc_id": "d2",
                "content": "Python basics for beginners",
                "title": "Doc 2",
            },
            {
                "doc_id": "d3",
                "content": "LangGraph reflection loop deep dive",
                "title": "Doc 3",
            },
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

    def test_hybrid_with_vector_results_keep_scores(self):
        """传入预计算 vector_results 时，RRF 双路融合生效，keep_scores=True 保留分数。"""
        docs = [
            {"doc_id": "d1", "content": "LangGraph reflection loop", "title": "A"},
            {"doc_id": "d2", "content": "BM25 keyword search", "title": "B"},
            {"doc_id": "d3", "content": "vector embedding retrieval", "title": "C"},
        ]
        # d1 同时在 BM25 和向量检索中排名靠前，RRF 分数应最高
        vector_results = [
            {
                "doc_id": "d1",
                "title": "A",
                "snippet": "LangGraph reflection",
                "_vector_score": 0.95,
            },
            {
                "doc_id": "d3",
                "title": "C",
                "snippet": "vector embedding",
                "_vector_score": 0.80,
            },
        ]
        results = hybrid_search(
            "LangGraph reflection",
            docs,
            k_vector=2,
            k_bm25=3,
            rrf_k=60,
            vector_results=vector_results,
            keep_scores=True,
        )
        # d1 双路命中，RRF 分数应最高
        assert results[0]["doc_id"] == "d1"
        assert "_rrf_score" in results[0]
        assert results[0]["_rrf_score"] > 0

    def test_hybrid_vector_results_strips_scores_by_default(self):
        """keep_scores=False（默认）时，内部字段被清理。"""
        docs = [
            {"doc_id": "d1", "content": "LangGraph reflection loop", "title": "A"},
            {"doc_id": "d3", "content": "vector embedding retrieval", "title": "C"},
        ]
        vector_results = [
            {"doc_id": "d1", "title": "A", "snippet": "x", "_vector_score": 0.9},
        ]
        results = hybrid_search(
            "LangGraph",
            docs,
            k_vector=1,
            k_bm25=2,
            rrf_k=60,
            vector_results=vector_results,
        )
        for r in results:
            assert "_rrf_score" not in r
            assert "_vector_score" not in r

    def test_vector_results_none_falls_back_to_bm25(self):
        """vector_results=None 时行为与旧版一致（BM25-only）。"""
        # 至少 3 条文档才能让 rank_bm25 IDF 对单一词产生正分
        docs = [
            {"doc_id": "d1", "content": "Python programming basics", "title": "A"},
            {
                "doc_id": "d2",
                "content": "LangGraph reflection loop implementation",
                "title": "B",
            },
            {
                "doc_id": "d3",
                "content": "Java concurrent programming patterns",
                "title": "C",
            },
            {"doc_id": "d4", "content": "Rust memory safety guide", "title": "D"},
        ]
        results = hybrid_search(
            "reflection", docs, k_bm25=4, k_vector=5, vector_results=None
        )
        assert len(results) >= 1
        assert results[0]["doc_id"] == "d2"
        for r in results:
            assert "_rrf_score" not in r
