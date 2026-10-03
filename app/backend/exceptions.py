"""统一异常体系与 FastAPI 全局异常处理器。"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend.exceptions")


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器，统一错误响应格式。"""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        trace_id = getattr(request.state, "trace_id", "unknown")
        logger.warning("[trace=%s] ValueError: %s", trace_id, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": str(exc),
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        trace_id = getattr(request.state, "trace_id", "unknown")
        logger.error("[trace=%s] RuntimeError: %s", trace_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "服务内部错误",
            },
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", "unknown")
        logger.exception("[trace=%s] Unhandled exception: %s", trace_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "服务内部错误",
            },
        )
