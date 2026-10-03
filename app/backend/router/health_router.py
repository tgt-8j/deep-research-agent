"""健康检查端点：/health（存活） + /health/ready（就绪）。"""
import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])
logger = logging.getLogger("backend.health")

# 应用启动时间
_start_time: float = 0


def set_start_time(t: float) -> None:
    global _start_time
    _start_time = t


class HealthResponse(BaseModel):
    status: str
    service: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health(service: str = "deepresearch-backend") -> HealthResponse:
    """存活探针：服务是否在运行。"""
    return HealthResponse(
        status="ok",
        service=service,
        uptime_seconds=time.time() - _start_time,
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """就绪探针：所有依赖是否可用。

    当前阶段仅检查 Python 进程状态，后续可扩展为：
    - 数据库连接池测试
    - Redis 连通性检测
    - Milvus 集合存在性检查
    """
    checks = {"process": True}
    # 可扩展：添加更多依赖检查
    all_ok = all(checks.values())
    return ReadyResponse(
        status="ok" if all_ok else "degraded",
        checks=checks,
    )
