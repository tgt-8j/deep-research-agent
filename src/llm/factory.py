"""LLM 工厂：根据配置创建 LLMAdapter 实例。

参考 shopkeeper-agent 的 llm/factory.py 和 llm/config.py。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .adapter import LLMAdapter, ModelConfig

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


def create_default_llm() -> LLMAdapter:
    """根据环境变量创建默认 LLMAdapter。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    model = os.getenv("MODEL", "qwen-plus")
    base_url = os.getenv(
        "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return LLMAdapter(
        config=ModelConfig(
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )
    )


def create_llm_for_agent(
    model: str,
    api_key: str,
    temperature: float,
) -> LLMAdapter:
    """为特定 Agent 创建 LLMAdapter（可指定不同档位）。"""
    base_url = os.getenv(
        "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return LLMAdapter(
        config=ModelConfig(
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
    )
