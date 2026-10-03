"""LLM 适配器模块：统一不同 Provider 的调用接口。

参考 shopkeeper-agent 的 LLM 适配模式（openai_compat / anthropic），
本项目默认使用 DashScope（阿里通义千问）兼容 OpenAI 协议。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore


@dataclass(frozen=True)
class ModelConfig:
    """LLM 模型配置。"""

    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.3


class LLMAdapter:
    """统一 LLM 调用接口，屏蔽底层 Provider 差异。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client: Any | None = None

    def _get_client(self):
        """延迟初始化，避免模块导入时立即建立连接。"""
        if self._client is None and ChatOpenAI is not None:
            os.environ.setdefault("DASHSCOPE_API_KEY", self.config.api_key)
            self._client = ChatOpenAI(
                api_key=self.config.api_key,
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
            )
        return self._client

    def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool] | None = None,
    ) -> dict:
        """同步调用 LLM，返回标准格式的 result dict。"""
        client = self._get_client()
        if client is None:
            raise RuntimeError("LLM adapter not initialized (ChatOpenAI not available)")
        if tools:
            client = client.bind_tools(tools)
        response = client.invoke(messages)
        return {"messages": [response]}

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool] | None = None,
    ) -> dict:
        """异步调用 LLM。"""
        client = self._get_client()
        if client is None:
            raise RuntimeError("LLM adapter not initialized (ChatOpenAI not available)")
        if tools:
            client = client.bind_tools(tools)
        response = await client.ainvoke(messages)
        return {"messages": [response]}
