"""请求追踪中间件：为每次请求生成 trace_id 并注入响应头。"""
from __future__ import annotations

import logging
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("backend.trace")


class TraceMiddleware(BaseHTTPMiddleware):
    """为每个请求生成唯一 trace_id，并通过响应头回传。"""

    async def dispatch(self, request: Request, call_next):
        # 优先使用客户端提供的 trace_id（用于关联上游请求）
        trace_id = request.headers.get("X-Trace-Id")
        if not trace_id:
            trace_id = f"{request.client.host[:8] if request.client else 'local'}-{uuid.uuid4().hex[:8]}"

        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
