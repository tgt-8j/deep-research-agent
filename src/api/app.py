"""App 生命周期管理：负责启动时初始化依赖，关闭时释放资源。

参考 shopkeeper-agent 的 api/lifespan.py：
- 在 lifespan 中一次性初始化客户端管理器
- 将初始化好的服务注入到 request.state，供路由层通过 Depends 获取
- 注册 CORS、异常处理、请求追踪中间件
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import AppConfig
from ..core.exceptions import register_exception_handlers
from ..middleware.tracing import tracing_middleware
from ..middleware.rate_limit import rate_limit_middleware
from ..persistence import SessionStore, WorkflowCancellation
from ..services.workflow import WorkflowService

logger = logging.getLogger("research.app")


class AppDependencies:
    """应用级依赖容器，所有共享资源在此统一管理。

    参考 shopkeeper-agent 的 api/dependencies.py 模式：
    - 客户端管理器（全局单例，lifespan 初始化）
    - 会话存储（每个请求共享）
    - WorkflowService（每个请求独立实例，可注入 cancellation）
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._session_store: SessionStore | None = None
        self._workflow_service: WorkflowService | None = None

    @property
    def session_store(self) -> SessionStore:
        if self._session_store is None:
            self._session_store = SessionStore(storage_root="./data/sessions")
        return self._session_store

    @property
    def workflow_service(self) -> WorkflowService:
        if self._workflow_service is None:
            self._workflow_service = WorkflowService(self._config)
            self._workflow_service._ensure_initialized()
        return self._workflow_service

    def create_request_context(self, session_key: str, user_id: str) -> dict:
        """为每次请求创建独立的上下文（包含取消信号）。

        参考 Paper-Agent 的 WorkflowRuntimeContext 模式。
        """
        return {
            "session_key": session_key,
            "user_id": user_id,
            "cancellation": WorkflowCancellation(),
        }


def create_app(config: AppConfig | None = None, service: WorkflowService | None = None) -> FastAPI:
    """创建 FastAPI 应用实例，并注册生命周期钩子。

    Args:
        config: 应用配置，不传则从环境变量加载
        service: 已初始化的 WorkflowService（用于测试注入 mock）
    """
    if config is None:
        config = AppConfig.from_file()

    # 应用级依赖容器（单例）
    app_deps = AppDependencies(config)
    if service is not None:
        app_deps._workflow_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：启动时初始化，关闭时清理。

        参考 shopkeeper-agent 的 api/lifespan.py：
        启动阶段创建所有客户端，关闭阶段释放连接。
        """
        logger.info("启动 Deep Research Agent 服务...")
        # 启动阶段：预初始化所有依赖
        _ = app_deps.workflow_service
        _ = app_deps.session_store
        logger.info("服务初始化完成 | model=%s | milvus=%s",
                     config.model, "enabled" if config.enable_milvus else "disabled")
        yield
        # 关闭阶段：清理资源
        logger.info("服务关闭中...")

    app = FastAPI(
        title="Deep Research Agent API",
        description="基于 LangGraph 的多智能体深度调研系统",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 注册中间件（测试环境跳过，避免 ASGI 中间件类问题）
    import os
    if os.getenv("TESTING") != "1":
        app.add_middleware(tracing_middleware)
        app.add_middleware(rate_limit_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册全局异常处理器
    register_exception_handlers(app)

    # 注册 Metrics 端点（Prometheus）
    from ..metrics import create_metrics_endpoint
    app.add_api_route("/metrics", create_metrics_endpoint(), methods=["GET"])

    # 将依赖容器注入到 app.state
    app.state.deps = app_deps

    # 注册路由
    from .routers.research_router import router
    app.include_router(router)

    # 健康检查
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "2.0.0"}

    return app
