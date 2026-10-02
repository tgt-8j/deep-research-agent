"""统一异常处理模块。

参考 shopkeeper-agent 的最佳实践，提供：
- 统一的业务异常类（AppError）
- FastAPI 全局异常处理器
- 标准化的错误响应格式
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("research.exceptions")


class AppError(Exception):
    """应用层业务异常基类。

    所有业务异常都应继承此类，便于统一处理。

    Attributes:
        code: 业务错误码（如 "RESEARCH_NOT_FOUND"）
        status_code: HTTP 状态码
        detail: 错误详情
    """

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(f"[{code}] {message}")


class NotFoundError(AppError):
    """资源不存在。"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource} not found: {identifier}",
            status_code=404,
        )


class ValidationError(AppError):
    """参数校验失败。"""

    def __init__(self, field: str, message: str):
        super().__init__(
            code="VALIDATION_ERROR",
            message=f"Invalid {field}: {message}",
            status_code=400,
        )


class ResearchError(AppError):
    """研究执行异常。"""

    def __init__(self, node: str, message: str):
        super().__init__(
            code="RESEARCH_ERROR",
            message=f"Research failed at {node}: {message}",
            status_code=500,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """注册所有异常处理器到 FastAPI 应用。"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "AppError | code=%s | status=%d | path=%s",
            exc.code, exc.status_code, request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def universal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception | path=%s | error=%s",
            request.url.path, exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                }
            },
        )
