"""健康检查 schemas。"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    uptime_seconds: float = 0.0
