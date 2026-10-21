"""研究查询 API 路由。

参考 shopkeeper-agent 的 api/routers/query_router.py + api/dependencies.py：
- 路由层只处理请求体、依赖声明和响应类型
- 不直接创建 Repository 或执行图节点
- 通过 FastAPI Depends 注入共享依赖
- 支持同步和 SSE 流式两种响应模式
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from src.api.app import AppDependencies
from src.api.schemas.research import ResearchRequest, ResearchResponse
from src.persistence.cancellation import WorkflowCancellation
from src.persistence.session_store import SessionStore
from src.services.workflow import WorkflowService

logger = logging.getLogger("research.api")

router = APIRouter(prefix="/api/v1/research", tags=["research"])


# =====================================================================
# 依赖注入（参考 shopkeeper-agent 的 api/dependencies.py 模式）
# =====================================================================


def get_app_deps(request: Request) -> AppDependencies:
    """从 app.state 获取应用级依赖容器。

    FastAPI 会在每次请求时自动调用此函数，
    确保每个请求都能拿到正确的依赖实例。
    """
    deps = getattr(request.app.state, "deps", None)
    if deps is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="AppDependencies 未初始化")
    return deps


def get_workflow_service(request: Request) -> WorkflowService:
    """获取 WorkflowService 实例（懒加载）。"""
    deps = get_app_deps(request)
    return deps.workflow_service


def get_session_store(request: Request) -> SessionStore:
    """获取 SessionStore 实例。"""
    deps = get_app_deps(request)
    return deps.session_store


def _resolve_tenant(request: Request, payload: ResearchRequest) -> str:
    """解析租户 ID：优先使用认证中间件注入的 tenant。"""
    auth_tenant = getattr(request.state, "tenant_id", None)
    if auth_tenant:
        if payload.tenant_id != auth_tenant:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail={"error": "tenant_mismatch"},
            )
        return auth_tenant
    return payload.tenant_id


def _emit(event: dict) -> str:
    """给事件补时间戳，并序列化成 SSE 的 data 段。

    注意：这里必须返回 JSON 字符串。改造前直接把 dict 插值进 f-string，
    发出去的是 Python repr（单引号、True/None），前端 ``JSON.parse`` 会直接
    抛错、一条事件都收不到（App.vue 里就是按 JSON 解析的）。
    """
    event["timestamp"] = datetime.now().isoformat()
    return json.dumps(event, ensure_ascii=False, default=str)


# =====================================================================
# 路由定义
# =====================================================================


@router.post("/run", response_model=ResearchResponse)
async def run_research(
    request: Request,
    payload: ResearchRequest,
    service: WorkflowService = Depends(get_workflow_service),
    session_store: SessionStore = Depends(get_session_store),
) -> ResearchResponse:
    """同步执行深度调研，返回完整报告。

    POST /api/v1/research/run
    Body: ResearchRequest (Pydantic 校验)
    """
    tenant_id = _resolve_tenant(request, payload)
    thread_id = payload.thread_id

    # 持久化会话
    session_store.create_session(
        session_key=thread_id,
        query=payload.query,
        user_id=payload.user_id,
        tenant_id=tenant_id,
    )

    final = await service.run(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
    )

    # 更新会话状态
    session_store.update_session_status(thread_id, "completed", final[:200])

    return ResearchResponse(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
        final=final,
    )


@router.post("/stream")
async def stream_research(
    request: Request,
    payload: ResearchRequest,
    service: WorkflowService = Depends(get_workflow_service),
    session_store: SessionStore = Depends(get_session_store),
) -> StreamingResponse:
    """流式执行深度调研，通过 SSE 实时推送进度和结果。

    POST /api/v1/research/stream
    """
    tenant_id = _resolve_tenant(request, payload)
    thread_id = payload.thread_id

    # 创建独立的取消信号（每个请求一个）
    cancellation = WorkflowCancellation()

    # 注册到应用级依赖容器：取消接口按 session_key 找到这个信号
    app_deps = get_app_deps(request)
    app_deps.active_runs[thread_id] = cancellation

    # 持久化会话
    session_store.create_session(
        session_key=thread_id,
        query=payload.query,
        user_id=payload.user_id,
        tenant_id=tenant_id,
    )

    async def event_stream():
        """SSE 事件流。

        改造点（原实现会在后台线程里调 ``asyncio.get_event_loop()``，
        在 Python 3.10+ 的无 loop 线程上直接抛 RuntimeError，导致 /stream
        永远只有一条 status 事件、30 秒后报"SSE 流超时"）：

        - 不再起线程：工作流直接以 ``asyncio.create_task`` 跑在事件循环里
        - ``emit`` 用 ``call_soon_threadsafe`` 投递，跨线程调用也安全
        - 队列空闲时发 SSE 注释行做心跳，而不是直接判超时
        - 客户端断开时取消任务，不留孤儿协程
        """
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: dict) -> None:
            """把事件投递到队列（可从任意线程调用）。"""
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def runner() -> None:
            try:
                await service.stream_events(
                    query=payload.query,
                    user_id=payload.user_id,
                    thread_id=thread_id,
                    tenant_id=tenant_id,
                    emit=emit,
                    cancellation=cancellation,
                )
            except asyncio.CancelledError:
                # 收到停止请求（或客户端断开）：告诉前端这是取消，不是报错
                emit({"type": "cancelled", "message": "任务已取消"})
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("[stream] 工作流异常")
                emit({"type": "error", "message": str(exc)})
            finally:
                emit({"type": "__done__"})

        task = asyncio.create_task(runner())

        # 1. 发送开始事件
        accepted = _emit({"type": "status", "message": "任务已接收，正在初始化工作流"})
        yield f"data: {accepted}\n\n"

        # 2. 持续消费队列，转发 SSE 事件
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # 心跳：避免反向代理把空闲连接掐断
                    yield ": keep-alive\n\n"
                    continue
                if event.get("type") == "__done__":
                    break
                yield f"data: {_emit(event)}\n\n"
        finally:
            # 3. 客户端断开或流结束：先请求协作式取消，再取消任务兜底
            if not task.done():
                cancellation.request()
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            app_deps.active_runs.pop(thread_id, None)
            # 4. 更新会话状态
            session_store.update_session_status(thread_id, "completed")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_key}/cancel")
async def cancel_session(
    session_key: str,
    request: Request,
) -> dict:
    """请求取消正在运行的调研任务。

    协作式取消：信号在**下一个节点边界**生效 —— 当前节点跑完后丢弃它的
    结果再中止，不会把半截状态写进 checkpoint（见 src/graph/graph.py 的取消边界）。
    """
    deps = get_app_deps(request)
    cancellation = deps.active_runs.get(session_key)
    if cancellation is None:
        raise HTTPException(status_code=404, detail="该会话没有正在运行的任务")
    cancellation.request()
    logger.info("[cancel] 收到取消请求 | session=%s", session_key)
    return {"session_key": session_key, "status": "cancel_requested"}


@router.get("/sessions/{session_key}")
async def get_session(
    session_key: str,
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """查询会话状态和结果。"""
    session = session_store.get_session(session_key)
    if not session:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="会话不存在")
    events = session_store.get_events(session_key, limit=50)
    return {
        "session": session,
        "events": events[-20:],  # 最近 20 条事件
    }


@router.get("/sessions")
async def list_sessions(
    user_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> list[dict]:
    """列出用户的会话历史。"""
    return session_store.list_sessions(user_id, limit=20)


@router.post("/sessions/{session_key}/resume")
async def resume_session(
    session_key: str,
    session_store: SessionStore = Depends(get_session_store),
    service: WorkflowService = Depends(get_workflow_service),
) -> dict:
    """从 Checkpoint 恢复会话并继续执行。"""
    checkpoint = session_store.get_latest_checkpoint(session_key)
    if not checkpoint:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="无可用的 Checkpoint")

    # 恢复状态并继续执行
    cancellation = WorkflowCancellation()
    final = await service.resume(
        thread_id=session_key,
        checkpoint_state=checkpoint.state_snapshot,
        cancellation=cancellation,
    )

    # 更新会话状态
    session_store.update_session_status(session_key, "completed", final[:200])

    return {
        "session_key": session_key,
        "restored_from_stage": checkpoint.stage,
        "final": final,
        "intent": "multiagent",
    }
