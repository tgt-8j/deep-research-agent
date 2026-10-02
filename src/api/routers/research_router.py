"""研究查询 API 路由。

参考 shopkeeper-agent 的 api/routers/query_router.py + api/dependencies.py：
- 路由层只处理请求体、依赖声明和响应类型
- 不直接创建 Repository 或执行图节点
- 通过 FastAPI Depends 注入共享依赖
- 支持同步和 SSE 流式两种响应模式
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.responses import StreamingResponse

from src.persistence.session_store import SessionStore
from src.persistence.cancellation import WorkflowCancellation
from src.services.workflow import WorkflowService
from src.api.schemas.research import ResearchRequest, ResearchResponse, StreamResponse

logger = logging.getLogger("research.api")

router = APIRouter(prefix="/api/v1/research", tags=["research"])


# =====================================================================
# 依赖注入（参考 shopkeeper-agent 的 api/dependencies.py 模式）
# =====================================================================


def get_app_deps(request: Request) -> "AppDependencies":
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


def _emit(event: dict) -> dict:
    """给事件添加时间戳，供 SSE 使用。"""
    event["timestamp"] = datetime.now().isoformat()
    return event


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

    # 持久化会话
    session_store.create_session(
        session_key=thread_id,
        query=payload.query,
        user_id=payload.user_id,
        tenant_id=tenant_id,
    )

    async def event_stream():
        """SSE 事件流。"""
        # 1. 发送开始事件
        yield f"data: {_emit({'type': 'status', 'message': '任务已接收，正在初始化工作流'})}\n\n"

        # 2. 在后台线程执行工作流，通过队列转发事件
        import asyncio

        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            else:
                queue.put_nowait(event)

        def worker():
            try:
                asyncio.get_event_loop().run_until_complete(
                    service.stream_events(
                        query=payload.query,
                        user_id=payload.user_id,
                        thread_id=thread_id,
                        tenant_id=tenant_id,
                        emit=emit,
                        cancellation=cancellation,
                    )
                )
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})
            finally:
                emit({"type": "__done__"})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # 3. 持续消费队列，转发 SSE 事件
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                emit({"type": "error", "message": "SSE 流超时"})
                break
            if event.get("type") == "__done__":
                break
            yield f"data: {_emit(event)}\n\n"

        # 4. 更新会话状态（在流结束后）
        session_store.update_session_status(thread_id, "completed")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
