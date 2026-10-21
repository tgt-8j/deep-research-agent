"""取消机制的回归测试。

修掉的缺陷：
1. `_with_cancellation_boundary` 读的是 `kwargs["runtime"]`，而 LangGraph 从不传
   这个 key —— 取消信号永远是 None，取消功等于是死代码；
2. `WorkflowService._ensure_initialized()` 也没把 `cancellation` 传给
   `build_graph`，即便边界能读到也拿不到请求级信号；
3. `reflect_node` 在 `missing_gaps` 为空时 `Command(goto="analyze")` 不递增
   iteration，`analyze -> reflect -> analyze` 会一直空转到撞递归上限。

这里覆盖：上下文绑定与隔离、节点边界的前/后检查、真实图的中止、
取消接口、以及 reflect 的计数递增。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langgraph.types import Command

from src.graph.graph import _should_continue_research, _with_cancellation_boundary
from src.persistence.cancellation import (
    WorkflowCancellation,
    cancellation_scope,
    current_cancellation,
)


class _Runtime:
    """最小 runtime shim：节点只用到 ``runtime.context.get(...)``。"""

    def __init__(self, context: dict | None = None):
        self.context = context or {}


# ---------------------------------------------------------------------------
# 上下文绑定
# ---------------------------------------------------------------------------


class TestCancellationScope:
    def test_binds_and_restores(self):
        assert current_cancellation() is None
        cancellation = WorkflowCancellation()
        with cancellation_scope(cancellation):
            assert current_cancellation() is cancellation
        assert current_cancellation() is None

    def test_restores_even_on_exception(self):
        cancellation = WorkflowCancellation()
        with pytest.raises(RuntimeError):
            with cancellation_scope(cancellation):
                raise RuntimeError("boom")
        assert current_cancellation() is None

    async def test_concurrent_scopes_are_isolated(self):
        first, second = WorkflowCancellation(), WorkflowCancellation()

        async def worker(cancellation):
            with cancellation_scope(cancellation):
                await asyncio.sleep(0)  # 让出控制权，交给另一个任务
                return current_cancellation()

        seen_first, seen_second = await asyncio.gather(worker(first), worker(second))
        assert seen_first is first
        assert seen_second is second


# ---------------------------------------------------------------------------
# 节点边界
# ---------------------------------------------------------------------------


class TestCancellationBoundary:
    async def test_blocks_before_node_when_already_requested(self):
        calls: list[str] = []

        async def node(state, **kwargs):
            calls.append("ran")
            return {}

        cancellation = WorkflowCancellation()
        cancellation.request()
        guarded = _with_cancellation_boundary("intent", node, cancellation)

        with pytest.raises(asyncio.CancelledError):
            await guarded({})

        assert calls == []  # 节点体一次都没进

    async def test_discards_result_when_requested_during_node(self):
        cancellation = WorkflowCancellation()

        async def node(state, **kwargs):
            # 模拟"节点执行期间用户点了停止"
            cancellation.request()
            return {"final": "半截结果"}

        guarded = _with_cancellation_boundary("write", node, cancellation)

        with pytest.raises(asyncio.CancelledError):
            await guarded({})

    async def test_runs_normally_without_request(self):
        async def node(state, **kwargs):
            return {"ok": True}

        guarded = _with_cancellation_boundary("plan", node, WorkflowCancellation())
        assert await guarded({}) == {"ok": True}

    async def test_sync_node_is_supported(self):
        def node(state, **kwargs):  # 同步节点：结果同样要被 await 逻辑兼容
            return {"ok": True}

        guarded = _with_cancellation_boundary("plan", node, None)
        assert await guarded({}) == {"ok": True}

    async def test_contextvar_wins_over_default(self):
        calls: list[str] = []

        async def node(state, **kwargs):
            calls.append("ran")
            return {}

        default = WorkflowCancellation()  # 默认信号：未请求
        request_level = WorkflowCancellation()
        request_level.request()

        guarded = _with_cancellation_boundary("intent", node, default)
        with cancellation_scope(request_level):
            with pytest.raises(asyncio.CancelledError):
                await guarded({})
        # 默认信号没被请求，说明生效的是上下文里的那个
        assert default.is_requested() is False
        assert calls == []


# ---------------------------------------------------------------------------
# 真实图：取消能穿透 LangGraph 中止整次运行
# ---------------------------------------------------------------------------


class TestGraphCancellation:
    async def test_ainvoke_aborts_when_cancelled(self):
        from src.graph import build_graph
        from src.services.workflow import NodeCancelledError

        graph = build_graph(llm=None, kb_client=None, bocha_api_key="")
        cancellation = WorkflowCancellation()
        cancellation.request()

        config = {"configurable": {"thread_id": "cancel-before-start"}}
        # LangGraph 会把节点里抛出的 CancelledError 包成 NodeCancelledError；
        # 服务层会把它翻译回 CancelledError（见下一个测试类）。
        with cancellation_scope(cancellation):
            with pytest.raises((asyncio.CancelledError, NodeCancelledError)):
                await graph.ainvoke({"query": "会被立刻取消的问题"}, config)


class TestServiceTranslation:
    async def test_node_cancellation_becomes_cancelled_error(self):
        """服务边界对外只暴露 ``CancelledError``，不会漏出 NodeCancelledError。

        LangGraph 1.2+ 把节点里的 CancelledError 包成 NodeCancelledError（继承
        Exception，会被 ``except Exception`` 当普通错误上报）；1.0.x 的
        ``astream_events`` 则是静默吞掉，流直接枯竭。两条路径都必须在服务层
        翻译成 CancelledError，所以这里用真实的取消信号驱动，而不是只依赖
        节点抛异常。
        """
        from langgraph.graph import END, START, StateGraph

        from src.services.workflow import WorkflowService
        from src.state import ResearchState

        async def boom(state):
            raise asyncio.CancelledError()

        builder = StateGraph(ResearchState)
        builder.add_node("boom", boom)
        builder.add_edge(START, "boom")
        builder.add_edge("boom", END)

        service = WorkflowService(_config())
        service._graph = builder.compile()

        cancellation = WorkflowCancellation()
        cancellation.request()

        with pytest.raises(asyncio.CancelledError):
            async for _event in service._iterate_events(
                {"query": "q"},
                {"configurable": {"thread_id": "t-translate"}},
                cancellation,
            ):
                pass


# ---------------------------------------------------------------------------
# reflect 计数：保证反思循环是有界的
# ---------------------------------------------------------------------------


class TestReflectIteration:
    async def test_no_gaps_jump_increments_iteration(self):
        from src.graph.nodes.reflect_node import reflect_node

        state = {
            "query": "q",
            "iteration": 1,
            "missing_gaps": [],
            "supplementary_queries": [],
        }
        result = await reflect_node(state, _Runtime({"llm": None}))

        assert isinstance(result, Command)
        assert result.goto == "analyze"
        # 关键：跳转必须带上 +1，否则 analyze ↔ reflect 会空转
        assert result.update == {"iteration": 2}

    def test_routing_stops_at_max_iterations(self):
        base = {"needs_more_research": True, "max_iterations": 2}
        assert _should_continue_research({**base, "iteration": 0}) == "reflect"
        assert _should_continue_research({**base, "iteration": 1}) == "reflect"
        assert _should_continue_research({**base, "iteration": 2}) == "write"


# ---------------------------------------------------------------------------
# 取消接口 + 任务注册表
# ---------------------------------------------------------------------------


def _config():
    from src.config import AppConfig

    return AppConfig(
        api_key="test-key",
        model="test-model",
        thread_id="default",
        user_id="default_user",
        tenant_id="default_tenant",
        max_iterations=1,
        enable_memory=False,
        milvus_host="127.0.0.1",
        milvus_port=19530,
        milvus_collection="test_collection",
        enable_milvus=False,
    )


def _build_app():
    from src.api.app import create_app

    return create_app(config=_config(), service=_FakeService())


class _FakeService:
    """只记录事件，不跑图。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.registered_at_start: set[str] = set()

    async def stream_events(
        self, query, user_id, thread_id, tenant_id, emit, cancellation=None
    ):
        emit({"type": "progress", "node": "intent", "status": "running"})

    async def run(self, **kwargs):  # pragma: no cover - 本文件不走 /run
        return ""


class _FakeSessionStore:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, str]] = []

    def create_session(self, **kwargs) -> None:
        pass

    def update_session_status(
        self, session_key: str, status: str, final_result: str | None = None
    ) -> None:
        self.status_updates.append((session_key, status))


class TestCancelEndpoint:
    def test_unknown_session_returns_404(self):
        from fastapi.testclient import TestClient

        app = _build_app()
        app.state.deps._session_store = _FakeSessionStore()
        with TestClient(app) as client:
            response = client.post("/api/v1/research/sessions/nope/cancel")
        assert response.status_code == 404

    def test_cancel_sets_flag_on_active_run(self):
        from fastapi.testclient import TestClient

        app = _build_app()
        app.state.deps._session_store = _FakeSessionStore()
        cancellation = WorkflowCancellation()
        app.state.deps.active_runs["t-42"] = cancellation

        with TestClient(app) as client:
            response = client.post("/api/v1/research/sessions/t-42/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "cancel_requested"
        assert cancellation.is_requested() is True

    def test_stream_registers_and_unregisters_run(self):
        from fastapi.testclient import TestClient

        app = _build_app()
        app.state.deps._session_store = _FakeSessionStore()
        observed: dict[str, set[str]] = {}

        original = app.state.deps._workflow_service.stream_events

        async def spy(*args, **kwargs):
            observed["during"] = set(app.state.deps.active_runs)
            await original(*args, **kwargs)

        app.state.deps._workflow_service.stream_events = spy

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/research/stream",
                json={"query": "q", "thread_id": "t-9"},
            ) as response:
                body = "".join(response.iter_text())

        events = [
            json.loads(line[len("data: ") :])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert any(event["type"] == "progress" for event in events)
        # 任务执行期间可被取消接口找到，结束后必须清理
        assert observed["during"] == {"t-9"}
        assert app.state.deps.active_runs == {}
