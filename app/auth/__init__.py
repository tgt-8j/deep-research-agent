"""JWT 认证模块：签发 Token + 验证中间件。

认证流程：
  1. POST /api/v1/auth/login  — 用 api_key 换取 JWT
  2. 客户端携带 Authorization: Bearer <token> 发起请求
  3. AuthMiddleware 验证 Token，将 user_id 注入 request.state
  4. 受保护路由从 request.state.user_id 读取身份

生产环境应将 SECRET_KEY 存入环境变量，不要硬编码。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger("auth")

# JWT 配置
SECRET_KEY = "deep-research-jwt-secret-key-change-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

# 内置租户 → API Key 映射
API_KEYS: dict[str, dict[str, str]] = {
    "tenant_demo": {"api_key": "demo_key", "user_id": "user_001", "name": "演示租户"},
}


class LoginRequest(BaseModel):
    api_key: str
    tenant_id: str | None = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    expires_in: int = TOKEN_EXPIRE_MINUTES


def _create_token(user_id: str, tenant_id: str) -> str:
    """签发 JWT Token。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant": tenant_id,
        "jti": uuid.uuid4().hex,
        "exp": now + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def login(request: LoginRequest) -> LoginResponse:
    """验证 API Key，返回 JWT。"""
    for tenant_id, info in API_KEYS.items():
        if info["api_key"] == request.api_key:
            token = _create_token(info["user_id"], tenant_id)
            logger.info("登录成功: tenant=%s user=%s", tenant_id, info["user_id"])
            return LoginResponse(
                access_token=token,
                user_id=info["user_id"],
                tenant_id=tenant_id,
            )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的 API Key 或租户 ID",
    )
