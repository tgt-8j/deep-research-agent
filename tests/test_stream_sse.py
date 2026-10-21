"""SSE 端点回归测试。

修复的 bug：``/api/v1/research/stream`` 原先把工作流丢进后台线程执行，并在
那个没有事件循环的线程里调用 ``asyncio.get_event_loop()``。Python 3.10+ 会
直接抛 ``RuntimeError``，线程随即死掉且不再投递 ``__done__``，客户端只能收到
一条 status 事件，然后干等 30 秒收到「SSE 流超时」。

本测试用不连 LLM 的假 service 验证真实路由 + SSE 管道：
1. 事件按序到达客户端；
2. 流正常结束（不是靠超时兜底）；
3. 结束后会话状态被更新为 completed。

如果换回"后台线程 + get_event_loop"的实现，本测试会挂。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import AppConfig


class _FakeService:
    """假的 WorkflowService：只吐固定事件，不碰网络/LLM。"""

    def __init__(self) -> None:
        self.stream_calls: list[dict] = []

    async def stream_events(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        emit,
        cancellation=None,
    ) -> None:
        self.stream_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "cancellation": cancellation,
            }
        )
        emit({"type": "progress", "node": "intent", "status": "running"})
        emit({"type": "delta", "content": "部分内容"})
        emit({"type": "final", "final": "# 研究报告"})

    async def run(self, **kwargs) -> str:  # pragma: no cover - 本测试不走 /run
        return "# 研究报告"


class _FakeSessionStore:
    """记录状态更新的假 SessionStore，避免测试写真实 sqlite。"""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.status_updates: list[tuple[str, str]] = []

    def create_session(self, **kwargs) -> None:
        self.created.append(kwargs)

    def update_session_status(
        self, session_key: str, status: str, final_result: str | None = None
    ) -> None:
        self.status_updates.append((session_key, status))


def _config() -> AppConfig:
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


def _build_client() -> tuple[TestClient, _FakeService, _FakeSessionStore]:
    fake_service = _FakeService()
    fake_store = _FakeSessionStore()
    app = create_app(config=_config(), service=fake_service)
    # 覆盖应用级依赖容器里的 session_store，避免测试往 ./data 写 sqlite
    app.state.deps._session_store = fake_store
    return TestClient(app), fake_service, fake_store


def test_stream_returns_events_and_completes():
    client, fake_service, fake_store = _build_client()

    with client.stream(
        "POST",
        "/api/v1/research/stream",
        json={"query": "LangGraph 是什么", "thread_id": "t-1"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]

    # 第一条必须是 status，且后续事件真的被推过来了（不是超时兜底）
    assert types[0] == "status"
    assert "progress" in types
    assert "delta" in types
    assert "final" in types
    assert "error" not in types

    # 工作流拿到的是请求里的 thread_id，且流结束后状态被置为 completed
    assert fake_service.stream_calls[0]["thread_id"] == "t-1"
    assert fake_store.status_updates == [("t-1", "completed")]


def test_stream_reports_workflow_error_without_hanging():
    """工作流内部抛异常时，应以 error 事件告知客户端并正常收尾。"""

    class _BoomService(_FakeService):
        async def stream_events(self, **kwargs) -> None:
            raise RuntimeError("boom")

    fake_store = _FakeSessionStore()
    app = create_app(config=_config(), service=_BoomService())
    app.state.deps._session_store = fake_store

    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/v1/research/stream", json={"query": "q"}
        ) as response:
            body = "".join(response.iter_text())

    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert any(
        event["type"] == "error" and "boom" in event.get("message", "")
        for event in events
    )
    assert fake_store.status_updates == [("default_thread", "completed")]
