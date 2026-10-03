"""FastAPI 应用工厂：注册路由、中间件与生命周期钩子。"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .backend.config import AppSettings
from .backend.exceptions import register_exception_handlers
from .backend.middleware.auth import AuthMiddleware
from .backend.middleware.trace import TraceMiddleware
from .backend.router import auth_router, health_router, research_router
from .logging_config import setup_logging
from .metrics import create_metrics_endpoint


def _create_app(settings: AppSettings) -> FastAPI:
    """创建 FastAPI 应用实例。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动：初始化共享资源
        logging.getLogger("backend").info("Application startup complete")
        yield
        # 关闭：清理资源（checkpointer、memory manager 等）
        logging.getLogger("backend").info("Application shutdown initiated")

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 注册中间件（顺序：auth → trace → cors）
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(TraceMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins(),
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Tenant-ID", "X-Trace-Id"],
    )

    # 注册全局异常处理器
    register_exception_handlers(app)

    # 注册路由
    app.include_router(health_router)
    app.include_router(research_router)
    app.include_router(auth_router.router)  # JWT 认证路由
    app.add_api_route("/metrics", create_metrics_endpoint(), methods=["GET"])  # Prometheus 指标

    return app


def create_app() -> FastAPI:
    settings = AppSettings()
    setup_logging(level="DEBUG" if settings.is_production else "INFO")
    return _create_app(settings)


# 模块级单例（供 uvicorn 直接引用）
app = create_app()


if __name__ == "__main__":
    runtime_settings = AppSettings()
    uvicorn.run(
        "app_main:app",
        host=runtime_settings.host,
        port=runtime_settings.port,
        reload=runtime_settings.app_env == "development",
    )
