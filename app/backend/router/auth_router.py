"""认证路由：登录接口。"""

from fastapi import APIRouter

from app.auth import LoginRequest, LoginResponse, login

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])


@router.post("/login", response_model=LoginResponse)
async def auth_login(request: LoginRequest) -> LoginResponse:
    """使用 api_key + tenant_id 换取 JWT Token。"""
    return await login(request)


@router.get("/keys")
async def list_valid_keys() -> dict:
    """返回可用的租户 Key 列表（开发调试用）。"""
    from app.auth import API_KEYS
    return {
        key: {"user_id": info["user_id"], "name": info["name"]}
        for key, info in API_KEYS.items()
    }
