import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..schemas import ResearchRequest, ResearchResponse
from ..service import WorkflowService, get_workflow_service


router = APIRouter(prefix="/api/v1/research", tags=["research"])


def _resolve_tenant(request: Request, payload: ResearchRequest) -> str:
    """解析租户 ID：优先使用认证中间件注入的 tenant，拒绝不匹配的请求。"""
    auth_tenant = getattr(request.state, "tenant_id", None)
    if auth_tenant:
        # 如果请求体中的 tenant 与认证租户不一致，拒绝
        if payload.tenant_id != auth_tenant:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail={"error": "tenant_mismatch", "message": "请求租户与认证租户不一致"},
            )
        return auth_tenant
    return payload.tenant_id


@router.post("/run", response_model=ResearchResponse)
async def run_research(
    request: Request,
    payload: ResearchRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ResearchResponse:
    tenant_id = _resolve_tenant(request, payload)
    final = await workflow_service.run(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=tenant_id,
        max_iterations=payload.max_iterations,
        enable_memory=payload.enable_memory,
    )
    return ResearchResponse(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=tenant_id,
        final=final,
    )


@router.post("/stream")
async def stream_research(
    request: Request,
    payload: ResearchRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> StreamingResponse:
    tenant_id = _resolve_tenant(request, payload)

    async def event_stream():
        start_event = {"type": "status", "message": "任务已接收，正在初始化多智能体链路"}
        yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
        async for event in workflow_service.stream_events(
            query=payload.query,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            tenant_id=tenant_id,
            max_iterations=payload.max_iterations,
            enable_memory=payload.enable_memory,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/async")
async def submit_research_async(
    request: Request,
    payload: ResearchRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> dict:
    """异步提交研究任务，返回 task_id。通过 GET /tasks/{task_id} 查询状态。"""
    tenant_id = _resolve_tenant(request, payload)
    import uuid
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # 提交异步任务到队列（由 TaskQueue 管理）
    from ..service.task_queue import task_queue
    coro = workflow_service.run(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=tenant_id,
        max_iterations=payload.max_iterations,
        enable_memory=payload.enable_memory,
    )
    await task_queue.submit(task_id, coro)

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> dict:
    """查询异步任务状态和结果。"""
    from ..service.task_queue import task_queue
    result = task_queue.get_status(task_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"error": "task_not_found"})
    return {
        "task_id": result.task_id,
        "status": result.status,
        "result": result.result,
        "error": result.error,
    }
