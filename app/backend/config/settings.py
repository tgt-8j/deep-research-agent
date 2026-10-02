from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_name: str = "DeepResearch Multi-Agent Assistant"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    config_path: str = str(Path(__file__).resolve().parents[3] / "config.json")

    # 租户 → API Key 映射（生产环境应通过环境变量或 KMS 加载）
    # 格式：TENANT_API_KEYS={"tenant1": "key1", "tenant2": "key2"}
    tenant_api_keys: dict[str, str] = {}

    # JWT 认证配置
    jwt_secret: str = "deep-research-jwt-secret-key-change-in-production"
    jwt_token_expire_minutes: int = 60

    # Agent 调用超时（秒）
    chat_timeout_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("tenant_api_keys", mode="before")
    @classmethod
    def parse_tenant_keys(cls, v):
        """支持 JSON 字符串或 dict 两种输入格式。"""
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # 兼容旧格式：逗号分隔的 key=value
                result = {}
                for pair in v.split(","):
                    if "=" in pair:
                        k, val = pair.split("=", 1)
                        result[k.strip()] = val.strip()
                return result
        return {}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def cors_origins(self) -> list[str]:
        values = [item.strip() for item in self.cors_allow_origins.split(",")]
        origins = [item for item in values if item]
        # 生产环境不允许通配符
        if self.is_production and "*" in origins:
            raise ValueError("生产环境不允许 CORS allow_origins 包含 *")
        return origins
