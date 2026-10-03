"""配置模块：统一管理应用配置。

参考 shopkeeper-agent 的 conf/ 目录，将配置从 mult_agents/config.py 迁移过来。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class AppConfig:
    """应用配置。

    字段说明：
    - api_key: DashScope API Key
    - model: 使用的模型名称
    - thread_id: 当前线程 ID（用于 LangGraph checkpointer）
    - user_id: 用户 ID
    - tenant_id: 租户 ID
    - max_iterations: 最大反思循环次数
    - enable_memory: 是否启用记忆功能
    - milvus_host/port/collection: Milvus 向量库配置
    """

    api_key: str
    model: str
    thread_id: str
    user_id: str
    tenant_id: str
    max_iterations: int
    enable_memory: bool
    milvus_host: str
    milvus_port: int
    milvus_collection: str
    enable_milvus: bool

    def with_overrides(self, **kwargs) -> AppConfig:
        """返回覆盖指定字段的新配置实例。"""
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **cleaned)

    @staticmethod
    def from_file(path: str | Path | None = None) -> AppConfig:
        """从 config.json 文件加载配置。"""
        config_path = Path(path) if path else _PROJECT_ROOT / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))

        def resolve_str(field: str, env_key: str, default: str = "") -> str:
            env_value = os.getenv(env_key)
            if env_value is not None and str(env_value).strip():
                return str(env_value).strip()
            file_value = data.get(field)
            if file_value is not None and str(file_value).strip():
                return str(file_value).strip()
            return default

        def resolve_int(field: str, env_key: str, default: int) -> int:
            return int(resolve_str(field, env_key, str(default)))

        def resolve_bool(field: str, env_key: str, default: bool) -> bool:
            return (
                resolve_str(field, env_key, "true" if default else "false").lower()
                == "true"
            )

        api_key = resolve_str("api_key", "DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 配置")

        return AppConfig(
            api_key=api_key,
            model=resolve_str("model", "MODEL", "qwen-plus"),
            thread_id=resolve_str("thread_id", "THREAD_ID", "default"),
            user_id=resolve_str("user_id", "USER_ID", "default_user"),
            tenant_id=resolve_str("tenant_id", "TENANT_ID", "default_tenant"),
            max_iterations=resolve_int("max_iterations", "MAX_ITERATIONS", 3),
            enable_memory=resolve_bool("enable_memory", "ENABLE_MEMORY", True),
            milvus_host=resolve_str("milvus_host", "MILVUS_HOST", "127.0.0.1"),
            milvus_port=resolve_int("milvus_port", "MILVUS_PORT", 19530),
            milvus_collection=resolve_str(
                "milvus_collection", "MILVUS_COLLECTION", "mult_agent_memory"
            ),
            enable_milvus=resolve_bool("enable_milvus", "ENABLE_MILVUS", True),
        )

    @staticmethod
    def from_env() -> AppConfig:
        """从环境变量加载配置（简化版）。"""
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量")
        return AppConfig(
            api_key=api_key,
            model=os.getenv("MODEL", "qwen-plus"),
            thread_id=os.getenv("THREAD_ID", "default"),
            user_id=os.getenv("USER_ID", "default_user"),
            tenant_id=os.getenv("TENANT_ID", "default_tenant"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            enable_memory=os.getenv("ENABLE_MEMORY", "true").lower() == "true",
            milvus_host=os.getenv("MILVUS_HOST", "127.0.0.1"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "mult_agent_memory"),
            enable_milvus=os.getenv("ENABLE_MILVUS", "true").lower() == "true",
        )
