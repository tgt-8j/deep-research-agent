"""Rerank 重排序模块测试。

验证 Reranker 类的粗筛、精排、融合逻辑，以及降级策略。
"""

from __future__ import annotations

import pytest

from src.retrieval.reranker import Reranker, rerank_evidence

# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evidence():
    """生成样本证据数据。"""
    return [
        {
            "source_id": "WEB-001",
            "source_type": "web",
            "title": "LangGraph 官方文档",
            "snippet": "LangGraph 是一个用于构建对话型 AI Agent 的框架",
            "url": "https://langchain.com/langgraph",
            "domain": "langchain.com",
        },
        {
            "source_id": "WEB-002",
            "source_type": "web",
            "title": "LangGraph 教程",
            "snippet": "LangGraph 支持状态管理和循环逻辑",
            "url": "https://example.com/tutorial",
            "domain": "example.com",
        },
        {
            "source_id": "LOCAL-001",
            "source_type": "local",
            "title": "内部知识库：Agent 架构",
            "snippet": "LangGraph reflection loop 实现原理分析",
            "doc_id": "doc_001",
            "domain": "",
        },
        {
            "source_id": "LOCAL-002",
            "source_type": "local",
            "title": "内部知识库：RAG 优化",
            "snippet": "检索增强生成的最佳实践",
            "doc_id": "doc_002",
            "domain": "",
        },
        {
            "source_id": "WEB-003",
            "source_type": "web",
            "title": "无关内容：Python 基础",
            "snippet": "Python 是一种通用编程语言",
            "url": "https://python.org",
            "domain": "python.org",
        },
    ]


# ---------------------------------------------------------------------------
# Reranker 初始化
# ---------------------------------------------------------------------------


class TestRerankerInit:
    def test_default_init(self):
        """默认初始化应启用 Embedding 重排序。"""
        reranker = Reranker()
        assert reranker.use_embedding_rerank is True

    def test_disable_embedding(self):
        """可以禁用 Embedding 重排序。"""
        reranker = Reranker(use_embedding_rerank=False)
        assert reranker.use_embedding_rerank is False


# ---------------------------------------------------------------------------
# 粗筛（coarse_filter）
# ---------------------------------------------------------------------------


class TestCoarseFilter:
    def test_empty_input(self):
        """空输入应返回空列表。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.coarse_filter("query", [], k=10)
        assert result == []

    def test_fewer_than_k(self, sample_evidence):
        """候选数少于 k 时应返回全部。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.coarse_filter("LangGraph", sample_evidence, k=10)
        assert len(result) == len(sample_evidence)

    def test_returns_dict_list(self, sample_evidence):
        """结果应为 dict 列表。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.coarse_filter("LangGraph", sample_evidence, k=3)
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], dict)


# ---------------------------------------------------------------------------
# 精排（fine_rank）
# ---------------------------------------------------------------------------


class TestFineRank:
    def test_empty_input(self):
        """空输入应返回空列表。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.fine_rank("query", [], k=10)
        assert result == []

    def test_fewer_than_k(self, sample_evidence):
        """候选数少于 k 时应返回全部。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.fine_rank("LangGraph", sample_evidence, k=10)
        assert len(result) == len(sample_evidence)

    def test_no_internal_fields(self, sample_evidence):
        """精排结果不应包含内部字段。"""
        reranker = Reranker(use_embedding_rerank=False)
        result = reranker.fine_rank("LangGraph", sample_evidence, k=3)
        for item in result:
            assert "_rerank_fine_score" not in item
            assert "_rerank_coarse_score" not in item


# ---------------------------------------------------------------------------
# 两级重排序（rerank）
# ---------------------------------------------------------------------------


class TestRerank:
    def test_empty_input(self):
        """空输入应返回空列表。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, stats = reranker.rerank("query", [])
        assert results == []
        assert stats["method"] == "none"

    def test_returns_tuple(self, sample_evidence):
        """应返回 (results, stats) 元组。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, stats = reranker.rerank("LangGraph", sample_evidence)
        assert isinstance(results, list)
        assert isinstance(stats, dict)

    def test_stats_structure(self, sample_evidence):
        """stats 应包含必要字段。"""
        reranker = Reranker(use_embedding_rerank=False)
        _, stats = reranker.rerank("LangGraph", sample_evidence)
        assert "method" in stats
        assert "total_before" in stats
        assert "total_after" in stats

    def test_source_type_preserved(self, sample_evidence):
        """source_type 字段应被保留。"""
        reranker = Reranker(use_embedding_rerank=False)
        results, _ = reranker.rerank("LangGraph", sample_evidence)
        source_types = {item.get("source_type") for item in results}
        # 原始数据有 web 和 local 两种类型
        assert "web" in source_types or "local" in source_types


# ---------------------------------------------------------------------------
# 便捷函数（rerank_evidence）
# ---------------------------------------------------------------------------


class TestRerankEvidence:
    def test_basic_call(self, sample_evidence):
        """便捷函数应正常工作。"""
        results, stats = rerank_evidence(
            "LangGraph", sample_evidence, coarse_k=3, fine_k=2
        )
        assert isinstance(results, list)
        assert isinstance(stats, dict)

    def test_with_custom_reranker(self, sample_evidence):
        """可传入自定义 Reranker 实例。"""
        custom_reranker = Reranker(use_embedding_rerank=False)
        results, _stats = rerank_evidence(
            "LangGraph",
            sample_evidence,
            coarse_k=3,
            fine_k=2,
            reranker=custom_reranker,
        )
        assert len(results) > 0


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_item(self):
        """单条证据应正常处理。"""
        evidence = [
            {
                "source_id": "X-001",
                "source_type": "web",
                "title": "Test",
                "snippet": "Content",
            }
        ]
        reranker = Reranker(use_embedding_rerank=False)
        results, _stats = reranker.rerank("query", evidence)
        assert len(results) == 1
        assert results[0]["source_id"] == "X-001"

    def test_missing_fields(self):
        """证据缺少某些字段时应能处理。"""
        evidence = [
            {"source_id": "X-001"},  # 缺少 title, snippet
            {"title": "Test"},  # 缺少 source_id
        ]
        reranker = Reranker(use_embedding_rerank=False)
        results, _ = reranker.rerank("query", evidence)
        assert len(results) == 2
