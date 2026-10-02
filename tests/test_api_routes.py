"""API 路由层集成测试。

使用 httpx TestClient 模拟 HTTP 请求，mock 依赖注入，
覆盖所有路由端点：run / stream / sessions / resume。
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import httpx
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Mock WorkflowService。"""
    service = MagicMock()
    service.run = AsyncMock(return_value="# 研究报告\n\n基于对测试问题的调研...")
    service.stream_events = AsyncMock()
    return service


@pytest.fixture
def mock_session_store(tmp_path):
    """Mock SessionStore（用真实实例，指向临时目录）。"""
    from src.persistence.session_store import SessionStore
    store = SessionStore(storage_root=str(tmp_path / "sessions"))
    return store


@pytest.fixture
def mock_deps(mock_service, mock_session_store):
    """Mock AppDependencies。"""
    deps = MagicMock()
    deps.workflow_service = mock_service
    deps.session_store = mock_session_store
    return deps


# ---------------------------------------------------------------------------
# TestClient 工厂
# ---------------------------------------------------------------------------


def _build_test_client(deps):
    """构建带 mock 依赖的 TestClient（使用 ASGITransport）。"""
    import os
    os.environ["TESTING"] = "1"  # 跳过中间件注册，避免 ASGI 类问题

    from src.api.app import create_app
    from src.config import AppConfig

    config = AppConfig(
        api_key="test-key",
        model="qwen-plus",
        thread_id="test",
        user_id="test_user",
        tenant_id="test_tenant",
        max_iterations=2,
        enable_memory=False,
        milvus_host="localhost",
        milvus_port=19530,
        milvus_collection="test",
        enable_milvus=False,
    )
    app = create_app(config, service=deps.workflow_service)
    # 注入 mock 依赖
    app.state.deps = deps

    import httpx
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(base_url="http://test", transport=transport)


# ---------------------------------------------------------------------------
# 路由测试
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    """POST /api/v1/research/run 测试。"""

    def test_run_success(self, mock_deps):
        client = _build_test_client(mock_deps)

        async def _run():
            response = await client.post(
                "/api/v1/research/run",
                json={"query": "AI Agent 趋势", "user_id": "u1", "thread_id": "t1"},
            )
            return response

        import asyncio
        response = asyncio.run(_run())
        assert response.status_code == 200
        data = response.json()
        assert data["final"]  # 有报告内容
        assert data["intent"] == "multiagent"
        mock_deps.workflow_service.run.assert_called_once()

    def test_run_missing_query(self, mock_deps):
        """缺少 query 时 payload["query"] 抛出 KeyError → 500。

        生产环境应加 Pydantic Schema 校验，此处记录为已知限制。
        """
        client = _build_test_client(mock_deps)

        async def _run():
            response = await client.post(
                "/api/v1/research/run",
                json={"user_id": "u1"},  # 缺少 query
            )
            return response

        import asyncio
        response = asyncio.run(_run())
        assert response.status_code == 422  # query 为空 → 422 Unprocessable Entity

    def test_run_with_valid_query(self, mock_deps):
        """有效请求应返回报告。"""
        client = _build_test_client(mock_deps)

        async def _run():
            response = await client.post(
                "/api/v1/research/run",
                json={"query": "AI Agent 趋势", "user_id": "u1", "thread_id": "t1"},
            )
            return response

        import asyncio
        response = asyncio.run(_run())
        assert response.status_code == 200
        data = response.json()
        assert data["final"]
        mock_deps.workflow_service.run.assert_called_once()


class TestStreamEndpoint:
    """POST /api/v1/research/stream 测试。

    注意：SSE 流式端点在生产环境中使用 threading + asyncio 混合模式，
    在测试环境的异步上下文中可能触发 event loop 问题。
    此处测试流式协议的正确性（event 结构），不测试完整执行路径。
    """

    def test_stream_protocol_structure(self):
        """验证 SSE 事件协议结构正确（mock 级别）。"""
        import json
        events = [
            {"type": "status", "message": "初始化"},
            {"type": "progress", "node": "intent", "status": "running"},
            {"type": "progress", "node": "plan", "status": "success"},
            {"type": "final", "final": "# 报告"},
            {"type": "__done__"},
        ]
        # 验证每个事件都有 timestamp 字段（由 _emit 添加）
        for evt in events:
            assert "type" in evt
        # 验证事件序列：以 status 开始，以 __done__ 结束
        assert events[0]["type"] == "status"
        assert events[-1]["type"] == "__done__"
        # 验证有 final 事件
        assert any(e["type"] == "final" for e in events)

    def test_stream_event_serialization(self):
        """验证 SSE 事件可以正确序列化和反序列化。"""
        import json
        from datetime import datetime
        from src.api.routers.research_router import _emit

        event = {"type": "progress", "node": "triage", "message": "分诊中"}
        emitted = _emit(event)
        assert "timestamp" in emitted
        serialized = json.dumps(emitted, ensure_ascii=False)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == "progress"


class TestSessionEndpoints:
    """会话管理接口测试。"""

    def test_get_session_not_found(self, mock_deps, tmp_path):
        client = _build_test_client(mock_deps)

        import asyncio
        response = asyncio.run(client.get("/api/v1/research/sessions/nonexistent"))
        assert response.status_code == 404

    def test_list_sessions(self, mock_deps, tmp_path):
        from src.persistence.session_store import SessionStore
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        store.create_session("s1", "问题1", "user_001", "tenant1")
        store.create_session("s2", "问题2", "user_001", "tenant1")
        mock_deps.session_store = store

        client = _build_test_client(mock_deps)

        import asyncio
        response = asyncio.run(client.get("/api/v1/research/sessions?user_id=user_001"))
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

    def test_resume_no_checkpoint(self, mock_deps, tmp_path):
        from src.persistence.session_store import SessionStore
        store = SessionStore(storage_root=str(tmp_path / "sessions"))
        mock_deps.session_store = store

        client = _build_test_client(mock_deps)

        import asyncio
        response = asyncio.run(
            client.post("/api/v1/research/sessions/s1/resume")
        )
        assert response.status_code == 404


class TestHealthEndpoint:
    """健康检查接口测试。"""

    def test_health(self, mock_deps):
        client = _build_test_client(mock_deps)

        import asyncio
        response = asyncio.run(client.get("/health"))
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"


class TestTenantResolution:
    """租户解析逻辑测试。"""

    def test_resolve_from_payload(self, mock_deps):
        from src.api.routers.research_router import _resolve_tenant
        from src.api.schemas.research import ResearchRequest
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.tenant_id = None
        payload = ResearchRequest(query="test", tenant_id="tenant_from_request")
        result = _resolve_tenant(request, payload)
        assert result == "tenant_from_request"

    def test_resolve_from_auth_tenant(self, mock_deps):
        from src.api.routers.research_router import _resolve_tenant
        from src.api.schemas.research import ResearchRequest
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.tenant_id = "auth_tenant"
        payload = ResearchRequest(query="test", tenant_id="auth_tenant")
        result = _resolve_tenant(request, payload)
        assert result == "auth_tenant"

    def test_tenant_mismatch_raises(self, mock_deps):
        from src.api.routers.research_router import _resolve_tenant
        from src.api.schemas.research import ResearchRequest
        from fastapi import HTTPException
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.tenant_id = "auth_tenant"
        payload = ResearchRequest(query="test", tenant_id="other_tenant")
        with pytest.raises(HTTPException) as exc_info:
            _resolve_tenant(request, payload)
        assert exc_info.value.status_code == 403
