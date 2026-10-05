"""低覆盖率模块测试补全：rerank, query_rewrite, local_rag, web_search node, rate_limit, tracing, config, main."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_runtime(llm=None, progress_emitter=None, kb_client=None):
    """创建兼容 graph 的 Mock Runtime。"""
    runtime = type("Runtime", (), {})()
    runtime.context = {
        "llm": llm,
        "kb_client": kb_client,
        "progress_emitter": progress_emitter,
        "retrieval_config": {"bocha_api_key": ""},
        "cancellation": None,
    }
    return runtime


def _make_state(**overrides):
    """创建默认 ResearchState 并应用覆盖。"""
    state = {
        "query": "测试问题",
        "sub_questions": ["子问题1"],
        "web_evidence": [],
        "local_evidence": [],
        "search_plan": [
            {
                "section_id": "sec_1",
                "query": "测试查询",
                "source_preference": "hybrid",
                "reason": "test",
            }
        ],
        "supplementary_queries": [],
        "iteration": 0,
        "messages": [],
        "web_retrieval_stats": {},
        "local_retrieval_stats": {},
        "web_search_trace": [],
        "local_rag_trace": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Rerank Node
# ---------------------------------------------------------------------------


class TestRerankNode:
    @pytest.mark.asyncio
    async def test_no_evidence(self):
        from src.graph.nodes.rerank_node import rerank_node

        state = _make_state(web_evidence=[], local_evidence=[])
        runtime = _make_runtime()
        result = await rerank_node(state, runtime)
        assert result["rerank_stats"]["method"] == "none"
        assert result["evidence_pool"] == []

    @pytest.mark.asyncio
    async def test_with_evidence_no_reranker(self):
        """Reranker 导入失败时降级。"""
        from src.graph.nodes.rerank_node import rerank_node

        evidence = [
            {"source_id": "WEB-1", "title": "证据1", "source_type": "web"},
            {"source_id": "WEB-2", "title": "证据2", "source_type": "web"},
        ]
        state = _make_state(web_evidence=evidence, local_evidence=[])
        runtime = _make_runtime()
        # Reranker 在函数内部 import，patch 模块级别
        with patch(
            "src.retrieval.reranker.Reranker", side_effect=ImportError("no module")
        ):
            result = await rerank_node(state, runtime)
        assert result["rerank_stats"]["method"] == "skipped_no_reranker"
        assert len(result["evidence_pool"]) == 2

    @pytest.mark.asyncio
    async def test_with_evidence_reranker_error(self):
        """Reranker 异常时降级返回原始顺序。"""
        from src.graph.nodes.rerank_node import rerank_node

        evidence = [{"source_id": "WEB-1", "title": "证据1", "source_type": "web"}]
        state = _make_state(web_evidence=evidence, local_evidence=[])
        runtime = _make_runtime()
        mock_instance = MagicMock()
        mock_instance.rerank.side_effect = RuntimeError("model not loaded")
        with patch("src.retrieval.reranker.Reranker", return_value=mock_instance):
            result = await rerank_node(state, runtime)
        assert result["rerank_stats"]["method"] == "fallback"
        assert len(result["evidence_pool"]) == 1

    @pytest.mark.asyncio
    async def test_merge_web_and_local(self):
        """合并 web 和 local 证据并传入 reranker。"""
        from src.graph.nodes.rerank_node import rerank_node

        web_ev = [{"source_id": "WEB-1", "source_type": "web"}]
        local_ev = [{"source_id": "LOC-1", "source_type": "local"}]
        state = _make_state(web_evidence=web_ev, local_evidence=local_ev)
        runtime = _make_runtime()
        mock_instance = MagicMock()
        mock_instance.rerank.return_value = (
            [{"source_id": "LOC-1"}, {"source_id": "WEB-1"}],
            {"method": "embedding", "coarse": 2, "fine": 2},
        )
        with patch("src.retrieval.reranker.Reranker", return_value=mock_instance):
            result = await rerank_node(state, runtime)
        assert result["rerank_stats"]["method"] == "embedding"
        assert len(result["evidence_pool"]) == 2
        call_args = mock_instance.rerank.call_args
        assert call_args[1]["evidence_items"] == web_ev + local_ev

    @pytest.mark.asyncio
    async def test_adds_missing_source_type(self):
        """缺失 source_type 的条目补为 unknown。"""
        from src.graph.nodes.rerank_node import rerank_node

        evidence = [{"source_id": "WEB-1", "title": "x"}]
        state = _make_state(web_evidence=evidence)
        runtime = _make_runtime()
        mock_instance = MagicMock()
        mock_instance.rerank.return_value = (evidence, {"method": "embedding"})
        with patch("src.retrieval.reranker.Reranker", return_value=mock_instance):
            result = await rerank_node(state, runtime)
        assert result["evidence_pool"][0]["source_type"] == "unknown"


# ---------------------------------------------------------------------------
# Query Rewrite Node
# ---------------------------------------------------------------------------


class TestQueryRewriteNode:
    @pytest.mark.asyncio
    async def test_llm_none_uses_fallback(self):
        from src.graph.nodes.query_rewrite_node import query_rewrite_node

        state = _make_state(query="LangGraph 是什么")
        runtime = _make_runtime(llm=None)
        result = await query_rewrite_node(state, runtime)
        assert len(result["rewritten_queries"]) > 0
        assert all(
            isinstance(q, str) and len(q) > 0 for q in result["rewritten_queries"]
        )

    @pytest.mark.asyncio
    async def test_llm_success_parses_json(self):
        from src.graph.nodes.query_rewrite_node import query_rewrite_node

        state = _make_state(query="Python asyncio 原理")
        llm = AsyncMock()
        llm.ainvoke.return_value = {
            "messages": [
                type(
                    "Msg",
                    (),
                    {
                        "content": '{"rewritten_queries": ["Python async", "asyncio tutorial"]}'
                    },
                )()
            ]
        }
        runtime = _make_runtime(llm=llm)
        result = await query_rewrite_node(state, runtime)
        assert len(result["rewritten_queries"]) == 2

    @pytest.mark.asyncio
    async def test_llm_json_parse_fails_uses_fallback(self):
        from src.graph.nodes.query_rewrite_node import query_rewrite_node

        state = _make_state(query="测试问题")
        llm = AsyncMock()
        llm.ainvoke.return_value = {
            "messages": [type("Msg", (), {"content": "not json at all"})()]
        }
        runtime = _make_runtime(llm=llm)
        result = await query_rewrite_node(state, runtime)
        assert len(result["rewritten_queries"]) > 0

    @pytest.mark.asyncio
    async def test_dedup_and_limit(self):
        from src.graph.nodes.query_rewrite_node import query_rewrite_node

        state = _make_state(query="LangGraph LangGraph LangGraph")
        runtime = _make_runtime(llm=None)
        result = await query_rewrite_node(state, runtime)
        assert len(result["rewritten_queries"]) <= 6
        assert len(set(result["rewritten_queries"])) == len(result["rewritten_queries"])

    @pytest.mark.asyncio
    async def test_progress_emitter_called(self):
        from src.graph.nodes.query_rewrite_node import query_rewrite_node

        state = _make_state(query="测试")
        progress_calls = []

        def progress(event_type, **kwargs):
            progress_calls.append({"event": event_type, **kwargs})

        runtime = _make_runtime(llm=None, progress_emitter=progress)
        await query_rewrite_node(state, runtime)
        assert any(c["event"] == "query_rewrite" for c in progress_calls)


# ---------------------------------------------------------------------------
# Local RAG Node
# ---------------------------------------------------------------------------


class TestLocalRagNode:
    @pytest.mark.asyncio
    async def test_no_kb_client(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        state = _make_state()
        runtime = _make_runtime(kb_client=None)
        result = await local_rag_node(state, runtime)
        assert result["local_evidence"] == []
        assert result["local_retrieval_stats"] == {}

    @pytest.mark.asyncio
    async def test_kb_client_without_hybrid_search(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        mock_kb = MagicMock(spec=["search"])
        mock_kb.search.return_value = [
            {"doc_id": "d1", "title": "文档1", "snippet": "内容1"}
        ]
        state = _make_state()
        runtime = _make_runtime(kb_client=mock_kb)
        result = await local_rag_node(state, runtime)
        assert len(result["local_evidence"]) == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_fallback(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        mock_kb = MagicMock()
        mock_kb.hybrid_search.side_effect = RuntimeError("milvus down")
        mock_kb.search.return_value = [
            {"doc_id": "d1", "title": "文档1", "snippet": "内容1"}
        ]
        state = _make_state()
        runtime = _make_runtime(kb_client=mock_kb)
        result = await local_rag_node(state, runtime)
        mock_kb.hybrid_search.assert_called_once()
        mock_kb.search.assert_called_once()
        assert len(result["local_evidence"]) == 1

    @pytest.mark.asyncio
    async def test_empty_records(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        mock_kb = MagicMock()
        mock_kb.search.return_value = []
        mock_kb.hybrid_search.return_value = []
        state = _make_state()
        runtime = _make_runtime(kb_client=mock_kb)
        result = await local_rag_node(state, runtime)
        assert result["local_evidence"] == []

    @pytest.mark.asyncio
    async def test_llm_none_uses_fallback_evidence(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        mock_kb = MagicMock()
        mock_kb.search.return_value = [{"doc_id": "d1", "title": "T1", "snippet": "S1"}]
        mock_kb.hybrid_search.return_value = []
        state = _make_state()
        runtime = _make_runtime(kb_client=mock_kb, llm=None)
        result = await local_rag_node(state, runtime)
        assert len(result["local_evidence"]) == 1
        ev = result["local_evidence"][0]
        assert ev["source_type"] == "local"
        assert ev["reliability_hint"] == "internal"

    @pytest.mark.asyncio
    async def test_deduplication(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        mock_kb = MagicMock()
        mock_kb.search.side_effect = [
            [{"doc_id": "d1", "title": "T1", "snippet": "S1"}],
            [{"doc_id": "d1", "title": "T1", "snippet": "S1"}],
        ]
        mock_kb.hybrid_search.return_value = []
        state = _make_state(
            search_plan=[
                {
                    "section_id": "sec_1",
                    "query": "q1",
                    "source_preference": "local",
                    "reason": "r",
                },
                {
                    "section_id": "sec_2",
                    "query": "q2",
                    "source_preference": "local",
                    "reason": "r",
                },
            ]
        )
        runtime = _make_runtime(kb_client=mock_kb, llm=None)
        result = await local_rag_node(state, runtime)
        assert len(result["local_evidence"]) == 1

    @pytest.mark.asyncio
    async def test_progress_emitter(self):
        from src.graph.nodes.local_rag_node import local_rag_node

        progress_calls = []

        def progress(event_type, **kwargs):
            progress_calls.append(event_type)

        mock_kb = MagicMock()
        mock_kb.search.return_value = []
        mock_kb.hybrid_search.return_value = []
        state = _make_state()
        runtime = _make_runtime(kb_client=mock_kb, progress_emitter=progress)
        await local_rag_node(state, runtime)
        assert "local_rag" in progress_calls


# ---------------------------------------------------------------------------
# Web Search Node
# ---------------------------------------------------------------------------


class TestWebSearchNode:
    @pytest.mark.asyncio
    async def test_empty_plan_uses_fallback(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state(search_plan=[])
        runtime = _make_runtime(llm=None)
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.return_value = []
            result = await web_search_node(state, runtime)
        assert result["web_evidence"] == []

    @pytest.mark.asyncio
    async def test_source_ids_assigned(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state(iteration=1)
        runtime = _make_runtime(llm=None)
        # 使用与查询相关的记录确保通过 _filter_records 过滤
        mock_record = {
            "title": "测试问题详解",
            "url": "https://example.com",
            "snippet": "关于测试问题的详细内容",
            "domain": "example.com",
        }
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.return_value = [mock_record]
            result = await web_search_node(state, runtime)
        assert len(result["web_evidence"]) >= 1
        assert result["web_evidence"][0]["source_id"].startswith("WEB2_")

    @pytest.mark.asyncio
    async def test_deduplication_by_url(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state(
            search_plan=[
                {
                    "section_id": "sec_1",
                    "query": "测试问题",
                    "source_preference": "web",
                    "reason": "",
                },
                {
                    "section_id": "sec_2",
                    "query": "测试问题",
                    "source_preference": "web",
                    "reason": "",
                },
            ]
        )
        runtime = _make_runtime(llm=None)
        # 使用官方域名确保通过 _filter_records
        same_record = {
            "title": "测试问题详解",
            "url": "https://example.gov.cn",
            "snippet": "关于测试问题的详细内容",
            "domain": "example.gov.cn",
        }
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.side_effect = [
                [same_record],
                [same_record],
            ]
            result = await web_search_node(state, runtime)
        assert len(result["web_evidence"]) == 1

    @pytest.mark.asyncio
    async def test_llm_none_fallback_evidence(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state()
        runtime = _make_runtime(llm=None)
        mock_record = {
            "title": "测试问题答案",
            "url": "https://example.com",
            "snippet": "这是关于测试问题的答案内容",
            "domain": "example.com",
        }
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.return_value = [mock_record]
            result = await web_search_node(state, runtime)
        assert len(result["web_evidence"]) >= 1
        ev = result["web_evidence"][0]
        assert ev["source_type"] == "web"
        assert ev["reliability_hint"] == "unknown"

    @pytest.mark.asyncio
    async def test_official_domain_detected(self):
        from src.retrieval import _is_official_domain

        assert _is_official_domain("example.gov.cn") is True
        assert _is_official_domain("example.edu.cn") is True
        assert _is_official_domain("example.com") is False

    @pytest.mark.asyncio
    async def test_evidence_capped_at_20(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state()
        runtime = _make_runtime(llm=None)
        big_records = [
            {
                "title": f"Article {i}",
                "url": f"https://ex{i}.com",
                "snippet": "s",
                "domain": "ex.com",
            }
            for i in range(30)
        ]
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.return_value = big_records
            result = await web_search_node(state, runtime)
        assert len(result["web_evidence"]) <= 20

    @pytest.mark.asyncio
    async def test_query_build_from_supplementary(self):
        from src.graph.nodes.web_search_node import web_search_node

        state = _make_state(
            iteration=1,
            supplementary_queries=[
                {
                    "section_id": "sec_gap",
                    "query": "gap_query",
                    "source_preference": "web",
                    "reason": "gap",
                }
            ],
            search_plan=[
                {
                    "section_id": "sec_1",
                    "query": "plan_q",
                    "source_preference": "web",
                    "reason": "",
                }
            ],
        )
        runtime = _make_runtime(llm=None)
        with patch.object(
            importlib.import_module("src.graph.nodes.web_search_node"),
            "bocha_web_search",
        ) as mock_search:
            mock_search.return_value = []
            await web_search_node(state, runtime)
            assert mock_search.call_args_list[0][0][0] == "gap_query"


# ---------------------------------------------------------------------------
# Rate Limit Middleware
# ---------------------------------------------------------------------------


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_allow_within_capacity(self):
        from src.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate=1.0, capacity=10)
        result = await bucket.acquire(1)
        assert result is True
        assert bucket._tokens == 9.0

    @pytest.mark.asyncio
    async def test_reject_when_empty(self):
        from src.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate=0.001, capacity=1)
        await bucket.acquire(1)
        result = await bucket.acquire(1)
        assert result is False

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        from src.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate=10.0, capacity=10)
        await bucket.acquire(10)
        assert await bucket.acquire(1) is False
        # 等待时间足够补充 token
        original_monotonic = bucket._last_refill
        bucket._last_refill = original_monotonic - 2.0  # 模拟过去了 2 秒
        result = await bucket.acquire(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_remaining(self):
        from src.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate=1.0, capacity=10)
        remaining = bucket.remaining()
        assert 0 <= remaining <= 10


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_first_request(self):
        from src.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(default_rate=2.0, default_capacity=10)
        assert await limiter.check("user_1") is True

    @pytest.mark.asyncio
    async def test_different_users_independent(self):
        from src.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(default_rate=2.0, default_capacity=1)
        assert await limiter.check("user_a") is True
        assert await limiter.check("user_b") is True

    @pytest.mark.asyncio
    async def test_exhausted_user_blocked(self):
        from src.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(default_rate=0.1, default_capacity=1)
        assert await limiter.check("user_x") is True
        assert await limiter.check("user_x") is False

    @pytest.mark.asyncio
    async def test_multi_token_acquire(self):
        from src.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(default_rate=1.0, default_capacity=5)
        assert await limiter.check("user_y", tokens=3) is True
        assert await limiter.check("user_y", tokens=3) is False


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_allows_normal_request(self):
        from src.middleware.rate_limit import rate_limit_middleware

        app = FastAPI()

        @app.post("/api/v1/research/run")
        async def run():
            return {"status": "ok"}

        # 使用 BaseHTTPMiddleware 包装，避免函数式中间件的签名问题
        from starlette.middleware.base import BaseHTTPMiddleware

        class MW(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await rate_limit_middleware(request, call_next)

        app.add_middleware(MW)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/research/run", json={"query": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_non_research_path_not_limited(self):
        from starlette.middleware.base import BaseHTTPMiddleware

        from src.middleware.rate_limit import rate_limit_middleware

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        class MW(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await rate_limit_middleware(request, call_next)

        app.add_middleware(MW)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tracing Middleware
# ---------------------------------------------------------------------------
# 注意：tracing_middleware 中 set_request_id 返回 str 而非 ContextVar token，
# 导致 token.reset() 在 Python 3.12 下触发 AttributeError。
# 因此不直接测试 ASGI 中间件，改为测试核心函数。
# ---------------------------------------------------------------------------


class TestTracingHelpers:
    """Tracing 辅助函数测试（绕过中间件的 bug）。"""

    def test_set_and_get_request_id(self):
        from src.core.log import get_request_id, set_request_id

        rid = set_request_id("my-trace-123")
        assert rid == "my-trace-123"
        assert get_request_id() == "my-trace-123"

    def test_default_request_id(self):
        from src.core.log import get_request_id

        rid = get_request_id()
        assert rid != "unknown"  # 有默认值
        assert len(rid) > 0

    def test_unique_ids(self):
        from src.core.log import get_request_id, set_request_id

        set_request_id("id-1")
        first = get_request_id()
        set_request_id("id-2")
        second = get_request_id()
        assert first == "id-1"
        assert second == "id-2"
        assert first != second

    def test_auto_generated_id(self):
        from src.core.log import get_request_id, set_request_id

        rid = set_request_id()  # 无参数，自动生成
        assert len(rid) == 8
        assert get_request_id() == rid


# ---------------------------------------------------------------------------
# Config Module
# ---------------------------------------------------------------------------


class TestConfigResolveHelpers:
    def test_from_env_missing_key_raises(self, monkeypatch):
        from src.config import AppConfig

        # 清除所有可能污染默认值的 env vars
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("ENABLE_MILVUS", raising=False)
        with pytest.raises(ValueError, match="缺少 DASHSCOPE_API_KEY"):
            AppConfig.from_env()
        # 确保环境变量恢复（monkeypatch 自动处理，但显式确保）
        import os as _os

        original_key = _os.environ.get("DASHSCOPE_API_KEY")
        if original_key:
            monkeypatch.setenv("DASHSCOPE_API_KEY", original_key)

    @pytest.mark.asyncio
    async def test_default_values(self, monkeypatch):
        from src.config import AppConfig

        # 清除所有可能污染默认值的 env vars
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("ENABLE_MILVUS", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        config = AppConfig.from_env()
        assert config.model == "qwen-plus"
        assert config.thread_id == "default"
        assert config.max_iterations == 3
        assert config.milvus_host == "127.0.0.1"
        assert config.milvus_port == 19530

    def test_override_values(self, monkeypatch):
        from src.config import AppConfig

        # 清除所有可能污染默认值的 env vars
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("ENABLE_MILVUS", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setenv("MODEL", "qwen-max")
        monkeypatch.setenv("MAX_ITERATIONS", "5")
        config = AppConfig.from_env()
        assert config.model == "qwen-max"
        assert config.max_iterations == 5

    def test_with_overrides(self, monkeypatch):
        from src.config import AppConfig

        # 清除所有可能污染默认值的 env vars
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("ENABLE_MILVUS", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        config = AppConfig.from_env()
        new_config = config.with_overrides(model="qwen-turbo", max_iterations=1)
        assert new_config.model == "qwen-turbo"
        assert new_config.max_iterations == 1
        assert new_config.api_key == config.api_key


class TestAppConfigFromFile:
    def test_file_not_found(self, tmp_path):
        from src.config import AppConfig

        with pytest.raises(FileNotFoundError):
            AppConfig.from_file(path=tmp_path / "nonexistent.json")

    @pytest.mark.asyncio
    async def test_load_from_json(self, tmp_path, monkeypatch):
        from src.config import AppConfig

        # 清除所有可能污染默认值的 env vars
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("ENABLE_MILVUS", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "api_key": "file-key",
                    "model": "qwen-max",
                    "max_iterations": 5,
                }
            )
        )
        config = AppConfig.from_file(path=config_file)
        assert config.api_key == "env-key"  # env 优先于文件
        assert config.model == "qwen-max"
        assert config.max_iterations == 5

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        from src.config import AppConfig

        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"api_key": "file-key", "model": "qwen-max"}))
        config = AppConfig.from_file(path=config_file)
        assert config.api_key == "env-key"


# ---------------------------------------------------------------------------
# Main Module
# ---------------------------------------------------------------------------


# Main app 测试已由 test_api_routes.py 覆盖，此处不再重复
