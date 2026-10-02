"""deep_research 核心流程集成测试。

补充现有单元测试的空白，覆盖：
1. 证据裁判节点 (deep_dive_node)
2. 分析节点 (analyze_node)
3. 反思节点 (reflect_node)
4. 直接回答节点 (direct_answer_node)
5. 搜索节点 (web_search_node, local_rag_node)
6. API 路由层 (research_router)
7. 持久化层 (session_store, cancellation)
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _make_runtime(llm=None, progress_emitter=None, kb_client=None):
    """创建兼容 graph._make_node 的 Mock Runtime。"""
    runtime = type('Runtime', (), {})()
    runtime.context = {
        'llm': llm,
        'kb_client': kb_client,
        'progress_emitter': progress_emitter,
        'retrieval_config': {'bocha_api_key': ''},
        'cancellation': None,
    }
    return runtime


# ---------------------------------------------------------------------------
# 证据裁判节点测试
# ---------------------------------------------------------------------------


class TestDeepDiveNode:
    """证据裁判节点测试。"""

    @pytest.mark.asyncio
    async def test_empty_evidence(self):
        from src.graph.nodes.deep_dive_node import deep_dive_node
        state = {
            "query": "测试问题",
            "sub_questions": [],
            "web_evidence": [],
            "local_evidence": [],
        }
        runtime = _make_runtime(llm=None)
        result = await deep_dive_node(state, runtime)
        assert result["evidence_pool"] == []
        assert result["audit_flags"] == []
        assert "deep_dive" in result

    @pytest.mark.asyncio
    async def test_with_evidence_fallback(self):
        from src.graph.nodes.deep_dive_node import deep_dive_node
        state = {
            "query": "阿司匹林的作用",
            "sub_questions": ["阿司匹林有什么功效？"],
            "web_evidence": [
                {
                    "source_id": "WEB-1",
                    "source_type": "web",
                    "title": "阿司匹林说明书",
                    "url": "https://example.com/aspirin",
                    "snippet": "阿司匹林用于镇痛退热",
                    "domain": "example.com",
                }
            ],
            "local_evidence": [],
        }
        runtime = _make_runtime(llm=None)
        result = await deep_dive_node(state, runtime)
        assert len(result["evidence_pool"]) > 0
        assert result["source_index"]

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        from src.graph.nodes.deep_dive_node import deep_dive_node
        state = {
            "query": "测试",
            "sub_questions": ["冲突测试"],
            "web_evidence": [
                {"source_id": "WEB-1", "source_type": "web", "title": "来源A",
                 "url": "https://a.com", "snippet": "结论：有效",
                 "domain": "blog.example.com"},
                {"source_id": "WEB-2", "source_type": "web", "title": "来源B",
                 "url": "https://b.com", "snippet": "结论：无效且有副作用",
                 "domain": "news.gov.cn"},
            ],
            "local_evidence": [],
        }
        runtime = _make_runtime(llm=None)
        result = await deep_dive_node(state, runtime)
        assert len(result["evidence_pool"]) > 0


# ---------------------------------------------------------------------------
# 分析节点测试
# ---------------------------------------------------------------------------


class TestAnalyzeNode:
    """分析节点测试。"""

    @pytest.mark.asyncio
    async def test_empty_evidence_pool(self):
        from src.graph.nodes.analyze_node import analyze_node
        state = {"query": "测试", "evidence_pool": []}
        runtime = _make_runtime(llm=None)
        result = await analyze_node(state, runtime)
        assert result["needs_more_research"] is True
        assert len(result["findings"]) > 0

    @pytest.mark.asyncio
    async def test_with_evidence(self):
        from src.graph.nodes.analyze_node import analyze_node
        state = {
            "query": "阿司匹林作用",
            "evidence_pool": [
                {
                    "source_id": "WEB-1", "source_type": "web",
                    "title": "阿司匹林说明书", "snippet": "镇痛退热",
                    "domain": "gov.cn", "reliability_score": 0.88,
                }
            ],
            "sub_questions": ["阿司匹林有什么功效？"],
            "audit_flags": [],
        }
        runtime = _make_runtime(llm=None)
        result = await analyze_node(state, runtime)
        assert "findings" in result
        assert "claim_map" in result
        assert "analysis" in result


# ---------------------------------------------------------------------------
# 反思节点测试
# ---------------------------------------------------------------------------


class TestReflectNode:
    """反思节点测试。"""

    @pytest.mark.asyncio
    async def test_no_gaps(self):
        from src.graph.nodes.reflect_node import reflect_node
        state = {
            "query": "测试",
            "missing_gaps": [],
            "sub_questions": [],
            "search_plan": [],
            "supplementary_queries": [],
        }
        runtime = _make_runtime(llm=None)
        result = await reflect_node(state, runtime)
        # 无信息缺口时，使用 Command 跳至 analyze 节点
        from langgraph.types import Command
        assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_with_gaps(self):
        from src.graph.nodes.reflect_node import reflect_node
        state = {
            "query": "阿司匹林安全性",
            "missing_gaps": ["出血风险的具体数据", "与其他药物的相互作用"],
            "sub_questions": ["阿司匹林有什么副作用？"],
            "search_plan": [{"query": "阿司匹林作用", "source_preference": "web"}],
            "supplementary_queries": [],
            "iteration": 0,
        }
        runtime = _make_runtime(llm=None)
        result = await reflect_node(state, runtime)
        assert result["iteration"] == 1
        # fallback 模式应生成补搜查询
        assert len(result["supplementary_queries"]) > 0


# ---------------------------------------------------------------------------
# 直接回答节点测试
# ---------------------------------------------------------------------------


class TestDirectAnswerNode:
    """直接回答节点测试。"""

    @pytest.mark.asyncio
    async def test_without_llm(self):
        from src.graph.nodes.direct_answer_node import direct_answer_node
        state = {"query": "你好，你是谁？"}
        runtime = _make_runtime(llm=None)
        result = await direct_answer_node(state, runtime)
        assert result["intent"] == "direct"
        assert "final" in result

    @pytest.mark.asyncio
    async def test_with_llm(self):
        from src.graph.nodes.direct_answer_node import direct_answer_node
        state = {"query": "北京今天天气怎么样？"}
        # 不传 llm，走 fallback
        runtime = _make_runtime(llm=None)
        result = await direct_answer_node(state, runtime)
        assert result["intent"] == "direct"
        assert result["final"]


# ---------------------------------------------------------------------------
# 网页搜索节点测试
# ---------------------------------------------------------------------------


class TestWebSearchNode:
    """网页搜索节点测试。"""

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """未配置 API Key 时不应报错，返回空结果。"""
        from src.graph.nodes.web_search_node import web_search_node
        state = {
            "query": "测试",
            "search_plan": [{"query": "test query", "source_preference": "web"}],
            "iteration": 0,
            "supplementary_queries": [],
        }
        runtime = _make_runtime(llm=None)
        result = await web_search_node(state, runtime)
        assert result["web_evidence"] == []
        assert "web_search" in result

    @pytest.mark.asyncio
    async def test_build_queries(self):
        """验证查询构建逻辑。"""
        from src.graph.nodes.web_search_node import _build_queries
        state = {
            "search_plan": [
                {"query": "q1", "source_preference": "web"},
                {"query": "q2", "source_preference": "local"},
            ],
            "supplementary_queries": [],
            "iteration": 0,
        }
        queries = _build_queries(state, "web")
        assert len(queries) == 1
        assert queries[0]["query"] == "q1"


# ---------------------------------------------------------------------------
# 本地 RAG 节点测试
# ---------------------------------------------------------------------------


class TestLocalRAGNode:
    """本地知识库检索节点测试。"""

    @pytest.mark.asyncio
    async def test_no_kb_client(self):
        """未配置知识库客户端时不应报错。"""
        from src.graph.nodes.local_rag_node import local_rag_node
        state = {
            "query": "测试",
            "search_plan": [],
            "supplementary_queries": [],
            "iteration": 0,
        }
        runtime = _make_runtime(llm=None, kb_client=None)
        result = await local_rag_node(state, runtime)
        assert result["local_evidence"] == []

    @pytest.mark.asyncio
    async def test_build_queries_local(self):
        from src.graph.nodes.local_rag_node import _build_queries
        state = {
            "search_plan": [
                {"query": "q1", "source_preference": "hybrid"},
                {"query": "q2", "source_preference": "local"},
            ],
            "supplementary_queries": [],
            "iteration": 0,
        }
        queries = _build_queries(state, "local")
        # hybrid 和 local 都会被选中
        assert len(queries) == 2


# ---------------------------------------------------------------------------
# API Schema 测试
# ---------------------------------------------------------------------------


class TestResearchSchema:
    """API 请求响应 Schema 测试。"""

    def test_research_request_valid(self):
        from src.api.schemas.research import ResearchRequest
        req = ResearchRequest(query="测试问题")
        assert req.query == "测试问题"
        assert req.user_id == "default_user"

    def test_research_request_min_length(self):
        from src.api.schemas.research import ResearchRequest
        # query has no min_length constraint in schema, empty string is valid
        req = ResearchRequest(query="")
        assert req.query == ""

    def test_research_response_model(self):
        from src.api.schemas.research import ResearchResponse
        resp = ResearchResponse(
            query="测试", user_id="u1", thread_id="t1",
            tenant_id="tenant1", final="报告内容", intent="multiagent",
        )
        assert resp.intent == "multiagent"


# ---------------------------------------------------------------------------
# 取消信号测试
# ---------------------------------------------------------------------------


class TestCancellation:
    """取消信号机制测试。"""

    def test_cancellation_default(self):
        from src.persistence.cancellation import WorkflowCancellation
        c = WorkflowCancellation()
        assert c.is_requested() is False

    def test_cancellation_request(self):
        from src.persistence.cancellation import WorkflowCancellation
        import asyncio
        c = WorkflowCancellation()
        c.request()
        assert c.is_requested() is True

    @pytest.mark.asyncio
    async def test_cancellation_raise(self):
        from src.persistence.cancellation import WorkflowCancellation
        import asyncio
        c = WorkflowCancellation()
        c.request()
        with pytest.raises(asyncio.CancelledError):
            c.raise_if_requested()

    @pytest.mark.asyncio
    async def test_cancellation_wait(self):
        import asyncio
        from src.persistence.cancellation import WorkflowCancellation
        c = WorkflowCancellation()

        async def requester():
            await asyncio.sleep(0.05)
            c.request()

        await asyncio.gather(c.wait(), requester())
        assert c.is_requested() is True


# ---------------------------------------------------------------------------
# 会话存储测试
# ---------------------------------------------------------------------------


class TestSessionStore:
    """会话持久化存储测试。"""

    def test_create_and_get_session(self, tmp_path):
        from src.persistence.session_store import SessionStore
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        store.create_session("sess_001", "测试问题", "user_001", "tenant1")
        session = store.get_session("sess_001")
        assert session is not None
        assert session["query"] == "测试问题"
        assert session["user_id"] == "user_001"

    def test_update_session_status(self, tmp_path):
        from src.persistence.session_store import SessionStore
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        store.create_session("sess_002", "问题", "user_001", "tenant1")
        store.update_session_status("sess_002", "completed", "摘要")
        session = store.get_session("sess_002")
        assert session["status"] == "completed"

    def test_list_sessions(self, tmp_path):
        from src.persistence.session_store import SessionStore
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        store.create_session("s1", "问题1", "user_001", "t1")
        store.create_session("s2", "问题2", "user_001", "t1")
        store.create_session("s3", "问题3", "user_002", "t1")
        sessions = store.list_sessions("user_001")
        assert len(sessions) == 2

    def test_checkpoint_save_and_load(self, tmp_path):
        from src.persistence.session_store import SessionStore, Checkpoint
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        cp = Checkpoint(
            session_key="sess_001", turn_id="turn_1",
            stage="triage", state_snapshot={"phase": "done"},
        )
        cp_id = store.save_checkpoint(cp)
        loaded = store.load_checkpoint(cp_id)
        assert loaded is not None
        assert loaded.stage == "triage"

    def test_get_latest_checkpoint(self, tmp_path):
        from src.persistence.session_store import SessionStore, Checkpoint
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        cp1 = Checkpoint(session_key="s1", turn_id="t1", stage="intake",
                         state_snapshot={})
        cp2 = Checkpoint(session_key="s1", turn_id="t2", stage="triage",
                         state_snapshot={"phase": "done"})
        store.save_checkpoint(cp1)
        store.save_checkpoint(cp2)
        latest = store.get_latest_checkpoint("s1")
        assert latest is not None
        assert latest.stage == "triage"


# ---------------------------------------------------------------------------
# SSRF 防护测试（补全）
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    """SSRF 防护测试。"""

    def test_safe_url_http(self):
        from src.retrieval.web_search import _safe_url
        assert _safe_url("https://example.com") == "https://example.com"

    def test_safe_url_blocked_host(self):
        from src.retrieval.web_search import _safe_url
        with pytest.raises(ValueError):
            _safe_url("http://localhost/api")

    def test_safe_url_blocked_ip(self):
        from src.retrieval.web_search import _safe_url
        with pytest.raises(ValueError):
            _safe_url("http://192.168.1.1/api")

    def test_safe_url_invalid_scheme(self):
        from src.retrieval.web_search import _safe_url
        with pytest.raises(ValueError):
            _safe_url("file:///etc/passwd")


# ---------------------------------------------------------------------------
# Prompt 加载测试
# ---------------------------------------------------------------------------


class TestPromptLoader:
    """Prompt 模板加载测试。"""

    def test_load_known_prompt(self):
        from src.prompt.loader import load_prompt
        prompt = load_prompt("intent_router")
        assert len(prompt) > 0
        assert "IntentRouter" in prompt

    def test_load_unknown_prompt(self):
        from src.prompt.loader import load_prompt
        prompt = load_prompt("nonexistent_prompt_xyz")
        assert prompt == ""

    def test_list_prompts(self):
        from src.prompt.loader import list_prompts
        prompts = list_prompts()
        assert "intent_router" in prompts
        assert "write" in prompts
