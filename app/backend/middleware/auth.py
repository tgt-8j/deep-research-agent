"""API 认证中间件：双层认证（JWT Bearer Token + API Key）。

认证优先级：
  1. JWT Bearer Token（Authorization: Bearer <token>）
     → 验证 Token，注入 request.state.user_id + tenant_id
  2. API Key + Tenant ID（X-API-Key + X-Tenant-ID header）
     → 验证 API Key，注入 request.state.tenant_id
  3. 两者都失败 → 返回 401

公开端点白名单无需认证。
"""
from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config.settings import AppSettings

logger = logging.getLogger("backend.auth")


# 公开端点白名单（无需认证）
_PUBLIC_PATHS = {
    "/health",
    "/api/v1/health",
    "/api/v1/health/ready",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    "/api/v1/auth/login",
    "/api/v1/auth/keys",
    "/metrics",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """双层认证中间件：JWT Bearer Token 优先，API Key 降级。"""

    def __init__(self, app, settings: AppSettings):
        super().__init__(app)
        self._settings = settings
        self._tenant_keys = settings.tenant_api_keys

    async def dispatch(self, request: Request, call_next):
        # 跳过公开端点
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/health"):
            return await call_next(request)

        # 尝试 JWT Bearer Token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]
            try:
                import jwt
                payload = jwt.decode(
                    token,
                    self._settings.jwt_secret,
                    algorithms=["HS256"]
                )
                request.state.user_id = payload["sub"]
                request.state.tenant_id = payload["tenant"]
                logger.debug("JWT 认证成功: user=%s tenant=%s", payload["sub"], payload["tenant"])
                return await call_next(request)
            except Exception as exc:
                logger.warning("JWT 认证失败，回退到 API Key: %s", exc)

        # 尝试 API Key + Tenant ID
        api_key = request.headers.get("X-API-Key")
        tenant_id = (
            request.query_params.get("tenant_id")
            or request.headers.get("X-Tenant-ID")
        )

        # 参数缺失 → 401
        if not api_key or not tenant_id:
            logger.warning("缺少认证信息: path=%s", request.url.path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "missing_credentials",
                    "message": "缺少认证信息，请使用 JWT Token 或 API Key + Tenant ID",
                },
            )

        # 校验租户 API Key
        stored_key = self._tenant_keys.get(tenant_id)
        if not stored_key or not _constant_time_compare(api_key, stored_key):
            logger.warning("认证失败: tenant=%s path=%s", tenant_id, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "invalid_credentials",
                    "message": "API Key 无效或租户不存在",
                },
            )

        # 认证成功：注入 request state
        request.state.tenant_id = tenant_id
        request.state.api_key = api_key
        request.state.user_id = None  # API Key 认证不设置 user_id
        return await call_next(request)


def _constant_time_compare(a: str, b: str) -> bool:
    """防时序攻击的字符串比较。"""
    import hmac
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
