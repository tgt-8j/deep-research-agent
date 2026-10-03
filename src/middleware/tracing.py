"""请求链路追踪中间件。

参考 Paper-Agent 的 ContextVar + loguru 模式：
- 为每个 HTTP 请求生成唯一的 request_id
- 将 request_id 注入到 ContextVar，供全链路日志使用
- 记录请求耗时和状态码
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response

from ..core.log import set_request_id

logger = logging.getLogger("research.middleware.tracing")


async def tracing_middleware(request: Request, call_next: Callable) -> Response:
    """为每个请求注入 request_id 并记录耗时。"""
    # 生成 request_id
    request_id = request.headers.get("x-request-id", str(time.time_ns()))
    token = set_request_id(request_id)

    try:
        # 记录请求开始
        start_time = time.perf_counter()
        logger.info(
            "REQUEST START | method=%s | path=%s | rid=%s",
            request.method,
            request.url.path,
            request_id,
        )

        # 执行请求
        response = await call_next(request)

        # 记录请求结束
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "REQUEST END  | method=%s | path=%s | status=%d | duration=%.1fms | rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        # 将 request_id 添加到响应头，方便前端追踪
        response.headers["x-request-id"] = request_id
        return response

    finally:
        # 清理 ContextVar
        token.reset()
