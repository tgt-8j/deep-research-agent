"""统一异常处理器测试"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.exceptions import register_exception_handlers


def _make_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-valueerror")
    async def raise_value():
        raise ValueError("bad input")

    @app.get("/raise-runtime")
    async def raise_runtime():
        raise RuntimeError("something went wrong")

    @app.get("/raise-generic")
    async def raise_generic():
        raise Exception("unexpected error")

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    return app


class TestExceptionHandlers:
    """全局异常处理器行为测试。"""

    def test_value_error_returns_400(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/raise-valueerror")
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "validation_error"
        assert "bad input" in data["message"]

    def test_runtime_error_returns_500(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/raise-runtime")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "internal_error"

    def test_unhandled_exception_returns_500(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/raise-generic")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "internal_error"

    def test_normal_request_unchanged(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
