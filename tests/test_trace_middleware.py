"""Trace 中间件测试"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.middleware.trace import TraceMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"trace_id": getattr(app.state, "trace_id", "none")}

    app.add_middleware(TraceMiddleware)
    return app


class TestTraceMiddleware:
    """Trace ID 注入和传递测试。"""

    def test_trace_id_in_response_header(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert "X-Trace-Id" in resp.headers
        assert len(resp.headers["X-Trace-Id"]) > 0

    def test_client_provided_trace_id(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test", headers={"X-Trace-Id": "my-custom-trace-123"})
        assert resp.status_code == 200
        assert resp.headers["X-Trace-Id"] == "my-custom-trace-123"

    def test_unique_trace_ids(self):
        """每次请求应生成不同的 trace_id。"""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp1 = client.get("/test")
        resp2 = client.get("/test")
        assert resp1.headers["X-Trace-Id"] != resp2.headers["X-Trace-Id"]
