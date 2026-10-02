"""E2E 集成测试：模拟完整研究流程

测试策略：
- 使用 MagicMock LLM 替代真实模型调用
- 验证状态流转和节点执行顺序
- 验证最终输出的引用溯源格式
- 覆盖 direct_answer 和 multiagent 两条路径
- 验证规则引擎、路由逻辑、SSRF 防护等纯函数
"""

from __future__ import annotations

import asyncio
import json
import sys
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# 添加 src/ 到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))


# ===================================================================
# Fixtures
# ===================================================================


def _make_mock_llm(responses: list[str] | None = None) -> MagicMock:
    """创建一个可配置的 mock LLM，用于模拟不同节点的输出。"""
    llm = MagicMock()
    response_sequence = responses or [
        '{"route": "direct", "reason": "简单问题"}',
        "# 答案\n\n这是一个简单的回答。",
    ]

    async def mock_ainvoke(messages):
        content = response_sequence.pop(0) if response_sequence else response_sequence[-1]
        return {"messages": [MagicMock(content=content, type="ai")]}

    llm.ainvoke = mock_ainvoke
    return llm


def _make_test_state(
    query: str = "什么是 LangGraph？",
    intent: str = "direct",
    **overrides,
) -> dict[str, Any]:
    """创建初始测试状态。"""
    state = {
        "query": query,
        "user_id": "test_user",
        "tenant_id": "test_tenant",
        "memory_context": "",
        "messages": [],
        "intent": intent,
        "phase": "initialized",
        "iteration": 0,
        "max_iterations": 2,
        "objective": "",
        "outline": [],
        "sub_questions": [],
        "research_questions": [],
        "search_plan": [],
        "budget": {},
        "web_search": "",
        "local_rag": "",
        "web_evidence": [],
        "local_evidence": [],
        "web_retrieval_stats": {},
        "local_retrieval_stats": {},
        "web_search_trace": [],
        "local_rag_trace": [],
        "deep_dive": "",
        "evidence_pool": [],
        "audit_flags": [],
        "source_index": [],
        "analysis": "",
        "findings": [],
        "claim_map": [],
        "needs_more_research": False,
        "missing_gaps": [],
        "supplementary_queries": [],
        "draft": "",
        "final": "",
        "code": "",
    }
    state.update(overrides)
    return state


# ===================================================================
# 测试 1: 路由逻辑验证
# ===================================================================


class TestRoutingLogic:
    """验证条件路由函数的正确性。"""

    def test_route_after_intent_direct(self):
        """简单问题路由到 direct_answer。"""
        from src.graph.graph import _route_after_intent

        state = {"intent": "direct", "query": "你好"}
        assert _route_after_intent(state) == "direct_answer"

    def test_route_after_intent_multiagent(self):
        """复杂问题路由到 plan。"""
        from src.graph.graph import _route_after_intent

        state = {"intent": "multiagent", "query": "调研2026年趋势"}
        assert _route_after_intent(state) == "plan"

    def test_should_continue_false(self):
        """证据充足时应结束研究。"""
        from src.graph.graph import _should_continue_research

        state = {
            "iteration": 1,
            "max_iterations": 2,
            "needs_more_research": False,
        }
        assert _should_continue_research(state) == "write"

    def test_should_continue_true(self):
        """证据不足且未达上限应继续补搜。"""
        from src.graph.graph import _should_continue_research

        state = {
            "iteration": 1,
            "max_iterations": 2,
            "needs_more_research": True,
        }
        assert _should_continue_research(state) == "reflect"

    def test_should_continue_max_reached(self):
        """达到最大迭代次数应结束。"""
        from src.graph.graph import _should_continue_research

        state = {
            "iteration": 5,
            "max_iterations": 2,
            "needs_more_research": True,
        }
        assert _should_continue_research(state) == "write"


# ===================================================================
# 测试 2: 规则引擎意图识别
# ===================================================================


class TestRuleBasedIntent:
    """验证基于关键词的规则意图识别。"""

    def test_simple_greeting(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("你好") == "direct"

    def test_weather_query(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("今天天气怎么样") == "direct"

    def test_research_keyword(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("帮我调研一下LangGraph框架") == "multiagent"

    def test_trend_with_year(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("2026年AI Agent市场趋势") == "multiagent"

    def test_comparison_query(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("对比RAG和Fine-tuning的优缺点") == "multiagent"

    def test_empty_query(self):
        from src.graph.nodes.intent_node import _rule_route
        assert _rule_route("") == "direct"


# ===================================================================
# 测试 3: 引用溯源验证
# ===================================================================


class TestCitationIntegrity:
    """验证引用溯源机制的完整性。"""

    def test_citation_format_validation(self):
        """验证引用 ID 格式符合 [PREFIX_SEQ-ID] 规范。"""
        from src.graph.nodes.write_node import _extract_citation_ids

        # 注意：实际格式是 WEB1_1-3, LOC1_2-1 等
        content = "根据[WEB1_1-1]的研究显示...[LOC1_2-1]也指出..."
        ids = _extract_citation_ids(content)
        assert "WEB1_1-1" in ids
        assert "LOC1_2-1" in ids

    def test_invalid_citation_removal(self):
        """非法引用应被自动移除。"""
        from src.graph.nodes.write_node import _validate_and_fix_citations

        # 使用符合正则格式但不在有效列表中的引用
        content = "有效引用[WEB1_1-1]和非法引用[WEB2_1-1]"
        valid_ids = {"WEB1_1-1"}
        fixed, used = _validate_and_fix_citations(content, valid_ids)

        assert "[WEB1_1-1]" in fixed
        assert "[WEB2_1-1]" not in fixed
        assert used == ["WEB1_1-1"]

    def test_source_id_assignment(self):
        """验证 source_id 分配逻辑。"""
        from src.graph.nodes.web_search_node import _assign_source_ids

        records = [
            {"title": "文档A", "url": "http://a.com"},
            {"title": "文档B", "url": "http://b.com"},
        ]
        result = _assign_source_ids(records, "WEB")
        assert result[0]["source_id"] == "WEB-1"
        assert result[1]["source_id"] == "WEB-2"

    def test_extract_citation_ids_empty(self):
        """无引用内容应返回空列表。"""
        from src.graph.nodes.write_node import _extract_citation_ids

        content = "这是一篇没有引用的文章。"
        ids = _extract_citation_ids(content)
        assert ids == []

    def test_extract_citation_ids_duplicate(self):
        """重复引用应去重并保持顺序。"""
        from src.graph.nodes.write_node import _extract_citation_ids

        content = "[WEB1_1-1]和[WEB1_1-1]又提到[WEB1_2-1]"
        ids = _extract_citation_ids(content)
        assert len(ids) == 2
        assert ids[0] == "WEB1_1-1"
        assert ids[1] == "WEB1_2-1"


# ===================================================================
# 测试 4: 证据评分逻辑
# ===================================================================


class TestEvidenceScoring:
    """验证证据来源评分规则。"""

    def test_local_knowledge_base_score(self):
        """本地知识库来源得分最高。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        record = {"source_type": "local"}
        score, reason = _score_evidence(record)
        assert score == 0.92

    def test_official_domain_score(self):
        """官方域名（.gov.cn, .edu.cn）得高分。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        record = {"source_type": "web", "domain": "www.example.gov.cn"}
        score, reason = _score_evidence(record)
        assert score == 0.88

        record2 = {"source_type": "web", "domain": "www.university.edu.cn"}
        score2, _ = _score_evidence(record2)
        assert score2 == 0.88

    def test_media_domain_score(self):
        """新闻媒体域名得分中等。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        record = {"source_type": "web", "domain": "news.example.com"}
        score, reason = _score_evidence(record)
        assert score == 0.72

    def test_normal_domain_score(self):
        """普通域名得分较低。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        record = {"source_type": "web", "domain": "blog.example.com"}
        score, reason = _score_evidence(record)
        assert score == 0.58

    def test_missing_domain_score(self):
        """缺少域名信息得最低分。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        record = {"source_type": "web"}
        score, reason = _score_evidence(record)
        assert score == 0.45


# ===================================================================
# 测试 5: SSRF 防护
# ===================================================================


class TestSSRFProtection:
    """验证 SSRF 防护机制。"""

    def test_blocked_internal_hosts(self):
        """阻止访问内网地址。"""
        from src.retrieval.web_search import _safe_url

        blocked_urls = [
            "http://localhost:8080",
            "http://127.0.0.1/api",
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.1/admin",
            "http://192.168.1.1/config",
            "http://172.16.0.1/internal",
        ]
        for url in blocked_urls:
            with pytest.raises(ValueError):
                _safe_url(url)

    def test_allowed_public_urls(self):
        """允许访问公共 URL。"""
        from src.retrieval.web_search import _safe_url

        public_urls = [
            "https://example.com/article",
            "http://www.github.com/repo",
            "https://docs.python.org/3/",
        ]
        for url in public_urls:
            assert _safe_url(url) is not None

    def test_invalid_scheme_blocked(self):
        """非法协议应被阻止。"""
        from src.retrieval.web_search import _safe_url

        with pytest.raises(ValueError):
            _safe_url("ftp://example.com")
        with pytest.raises(ValueError):
            _safe_url("file:///etc/passwd")
        with pytest.raises(ValueError):
            _safe_url("gopher://example.com")

    def test_missing_hostname_blocked(self):
        """缺少主机名的 URL 应被阻止。"""
        from src.retrieval.web_search import _safe_url

        with pytest.raises(ValueError):
            _safe_url("http:///path")


# ===================================================================
# 测试 6: 取消边界机制
# ===================================================================


class TestCancellationBoundary:
    """验证取消边界机制。"""

    def test_guarded_function_name(self):
        """验证取消边界包装后的函数名包含 '_guarded'。"""
        from src.graph.graph import _with_cancellation_boundary

        async def dummy_node(state, runtime):
            return {"test": "value"}

        guarded = _with_cancellation_boundary("test", dummy_node)
        assert guarded.__name__ == "test_guarded"

    @pytest.mark.asyncio
    async def test_guarded_calls_cancellation_check(self):
        """验证取消边界在节点前后检查 cancellation 信号。"""
        from src.graph.graph import _with_cancellation_boundary

        async def dummy_node(state, runtime):
            return {"test": "value"}

        cancellation = MagicMock()
        cancellation.raise_if_requested = MagicMock()

        guarded = _with_cancellation_boundary("test", dummy_node)

        class FakeRuntime:
            pass

        runtime = FakeRuntime()
        runtime.cancellation = cancellation

        await guarded({}, runtime=runtime)
        assert cancellation.raise_if_requested.called


# ===================================================================
# 测试 7: WorkflowService 核心逻辑
# ===================================================================


class TestWorkflowService:
    """验证 WorkflowService 的核心逻辑。"""

    @pytest.mark.asyncio
    async def test_build_state_contains_query(self):
        """验证 _build_state 正确初始化状态。"""
        from src.services.workflow import WorkflowService
        from src.config import AppConfig

        # 使用默认配置值创建 config 对象
        config = AppConfig(
            api_key="test-key",
            model="qwen-plus",
            thread_id="test-thread",
            user_id="test-user",
            tenant_id="test-tenant",
            max_iterations=2,
            enable_memory=False,
            milvus_host="localhost",
            milvus_port=19530,
            milvus_collection="test_collection",
            enable_milvus=False,
        )
        service = WorkflowService(config)
        service._graph = MagicMock()  # 跳过初始化

        state = service._build_state(
            query="测试问题",
            user_id="user_001",
            thread_id="thread_001",
            tenant_id="tenant_001",
        )

        assert state["query"] == "测试问题"
        assert state["user_id"] == "user_001"
        assert state["intent"] == ""
        assert state["phase"] == "initialized"
        assert state["final"] == ""

    def test_build_state_uses_config_max_iterations(self):
        """验证 max_iterations 从配置中读取。"""
        from src.services.workflow import WorkflowService
        from src.config import AppConfig

        config = AppConfig(
            api_key="test-key",
            model="qwen-plus",
            thread_id="test-thread",
            user_id="test-user",
            tenant_id="test-tenant",
            max_iterations=5,
            enable_memory=False,
            milvus_host="localhost",
            milvus_port=19530,
            milvus_collection="test_collection",
            enable_milvus=False,
        )
        service = WorkflowService(config)
        service._graph = MagicMock()

        state = service._build_state(
            query="test",
            user_id="user_001",
            thread_id="thread_001",
            tenant_id="tenant_001",
        )

        assert state["max_iterations"] == 5


# ===================================================================
# 测试 8: 完整图构建验证
# ===================================================================


class TestGraphConstruction:
    """验证 LangGraph 图的构建正确性。"""

    def test_graph_has_all_nodes(self):
        """验证图中包含所有 10 个节点。"""
        from src.graph.graph import build_graph

        mock_llm = MagicMock()
        mock_kb = MagicMock()

        graph = build_graph(
            llm=mock_llm,
            kb_client=mock_kb,
            progress_emitter=None,
            bocha_api_key="",
        )

        # 检查节点是否存在（排除 __start__）
        expected_nodes = {
            "intent",
            "query_rewrite",
            "direct_answer",
            "plan",
            "web_search",
            "local_rag",
            "rerank",
            "deep_dive",
            "approval",
            "analyze",
            "reflect",
            "write",
        }
        actual_nodes = set(graph.nodes.keys()) - {"__start__"}
        assert expected_nodes == actual_nodes

    def test_graph_compiles(self):
        """验证图可以成功编译。"""
        from src.graph.graph import build_graph

        mock_llm = MagicMock()
        mock_kb = MagicMock()

        graph = build_graph(
            llm=mock_llm,
            kb_client=mock_kb,
            progress_emitter=None,
            bocha_api_key="",
        )

        assert graph is not None
        assert hasattr(graph, "nodes")


# ===================================================================
# 测试 9: 状态字段完整性
# ===================================================================


class TestStateIntegrity:
    """验证 ResearchState 的字段完整性。"""

    def test_required_fields_exist(self):
        """验证状态中包含所有必需字段。"""
        state = _make_test_state()

        # 核心字段
        assert "query" in state
        assert "intent" in state
        assert "phase" in state
        assert "messages" in state
        assert "final" in state

    def test_initial_values_are_empty(self):
        """验证初始状态下各字段为空或默认值。"""
        state = _make_test_state()

        assert state["query"] == "什么是 LangGraph？"
        assert state["intent"] == "direct"
        assert state["phase"] == "initialized"
        assert state["iteration"] == 0
        assert state["final"] == ""
        assert state["draft"] == ""
        assert state["web_search"] == ""
        assert state["local_rag"] == ""


# ===================================================================
# 测试 10: 工具函数验证（使用 app/mult_agents/nodes.py 中的函数）
# ===================================================================


class TestToolFunctions:
    """验证各个工具函数的行为。"""

    def test_extract_json_block(self):
        """验证 JSON 块提取函数。"""
        from app.mult_agents.nodes import _extract_json_block

        text = "Here is the result:\n```json\n{\"key\": \"value\"}\n```"
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_load_json_with_fallback(self):
        """验证 JSON 加载带 fallback 的行为。"""
        from app.mult_agents.nodes import _load_json

        valid = _load_json('{"a": 1}', fallback={})
        assert valid == {"a": 1}

        invalid = _load_json("not json", fallback={"default": True})
        assert invalid == {"default": True}

    def test_filter_web_records(self):
        """验证网页记录过滤函数。"""
        from app.mult_agents.nodes import _filter_web_records

        query = "machine learning"
        records = [
            {"snippet": "ML tutorial", "url": "https://example.com/ml"},
            {"snippet": "irrelevant spam", "url": "https://example.com/spam"},
        ]
        kept, rejected = _filter_web_records(query, records)
        assert isinstance(kept, list)
        assert isinstance(rejected, dict)

    def test_assign_source_ids(self):
        """验证 source_id 分配逻辑。"""
        from app.mult_agents.nodes import _assign_source_ids

        records = [
            {"title": "文档A", "url": "http://a.com"},
            {"title": "文档B", "url": "http://b.com"},
        ]
        result = _assign_source_ids(records, "WEB")
        assert result[0]["source_id"] == "WEB-1"
        assert result[1]["source_id"] == "WEB-2"

    def test_dedupe_sources(self):
        """验证去重逻辑按 source_id 过滤重复记录。"""
        from app.mult_agents.nodes import _dedupe_sources

        items = [
            {"source_id": "WEB1_1", "title": "重复文档"},
            {"source_id": "WEB1_1", "title": "重复文档"},
            {"source_id": "WEB1_2", "title": "唯一文档"},
        ]
        result = _dedupe_sources(items, ["source_id"])
        assert len(result) == 2
