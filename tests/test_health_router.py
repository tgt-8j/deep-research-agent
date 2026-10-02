"""健康检查端点测试"""
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.router import health_router
from backend.router.health_router import set_start_time


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    set_start_time(0.0)  # 固定启动时间便于测试
    return app


class TestHealthEndpoint:
    """/health 存活探针测试。"""

    def test_health_returns_ok(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "deepresearch-backend"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_has_uptime(self):
        app = _make_app()
        set_start_time(time.time() - 100)  # 模拟运行 100 秒
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 99


class TestReadyEndpoint:
    """/health/ready 就绪探针测试。"""

    def test_ready_returns_ok(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["checks"]["process"] is True
