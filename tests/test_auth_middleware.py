"""认证中间件测试"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.config.settings import AppSettings
from backend.middleware.auth import AuthMiddleware


def _make_app(tenant_keys: dict) -> FastAPI:
    """创建带认证中间件的测试应用。"""
    app = FastAPI()

    @app.get("/protected")
    async def protected(request: Request):
        return {
            "tenant": request.state.tenant_id,
            "user": getattr(request.state, "user_id", None),
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    settings = AppSettings(tenant_api_keys=tenant_keys)
    app.add_middleware(AuthMiddleware, settings=settings)
    return app


class TestAuthMiddleware:
    """认证中间件核心行为测试。"""

    def test_valid_api_key(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/protected", headers={"X-API-Key": "key_a", "X-Tenant-ID": "tenant_a"}
        )
        assert resp.status_code == 200
        assert resp.json()["tenant"] == "tenant_a"
        assert resp.json()["user"] is None

    def test_missing_api_key(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/protected", headers={"X-Tenant-ID": "tenant_a"})
        assert resp.status_code == 401

    def test_missing_tenant_id(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/protected", headers={"X-API-Key": "key_a"})
        assert resp.status_code == 401

    def test_invalid_api_key(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/protected", headers={"X-API-Key": "wrong_key", "X-Tenant-ID": "tenant_a"}
        )
        assert resp.status_code == 403

    def test_unknown_tenant(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/protected",
            headers={"X-API-Key": "key_a", "X-Tenant-ID": "tenant_unknown"},
        )
        assert resp.status_code == 403

    def test_query_param_tenant(self):
        app = _make_app({"tenant_a": "key_a"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/protected?tenant_id=tenant_a", headers={"X-API-Key": "key_a"}
        )
        assert resp.status_code == 200


class TestJWTAuth:
    """JWT Bearer Token 认证测试。"""

    def _create_jwt(self, user_id: str, tenant_id: str) -> str:
        from datetime import datetime, timedelta, timezone

        import jwt

        SECRET_KEY = "deep-research-jwt-secret-key-change-in-production"
        payload = {
            "sub": user_id,
            "tenant": tenant_id,
            "jti": "test-jti",
            "exp": datetime.now(timezone.utc) + timedelta(days=1),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    def test_valid_jwt_token(self):
        app = _make_app({})
        client = TestClient(app, raise_server_exceptions=False)
        token = self._create_jwt("user_001", "tenant_admin")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user"] == "user_001"
        assert resp.json()["tenant"] == "tenant_admin"

    def test_expired_jwt_token(self):
        app = _make_app({})
        client = TestClient(app, raise_server_exceptions=False)
        from datetime import datetime, timedelta, timezone

        import jwt

        SECRET_KEY = "deep-research-jwt-secret-key-change-in-production"
        payload = {
            "sub": "user_001",
            "tenant": "tenant_admin",
            "jti": "test-expired",
            "exp": datetime.now(timezone.utc) - timedelta(days=1),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_invalid_jwt_token(self):
        app = _make_app({})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/protected", headers={"Authorization": "Bearer invalid.token.here"}
        )
        assert resp.status_code == 401


class TestPublicPaths:
    """公开端点无需认证。"""

    def test_health_no_auth(self):
        app = _make_app({})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
