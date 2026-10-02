"""多模型适配层。

参考 Paper-Agent 的 llm/ 目录，支持 OpenAI 兼容协议、Anthropic 协议、
以及 DashScope（通义千问）等多模型后端。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger("research.llm")


@dataclass
class ModelConfig:
    """LLM 模型配置。"""
    provider: str           # "openai" | "anthropic" | "dashscope"
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.3
    max_tokens: int = 4096


class LLMAdapter:
    """LLM 统一适配接口。

    所有模型后端都通过此接口调用，上层代码不感知具体 Provider。
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client: Optional[Any] = None

    def _get_client(self):
        """延迟初始化客户端。"""
        if self._client is not None:
            return self._client

        if self.config.provider in ("openai", "dashscope"):
            from langchain_openai import ChatOpenAI
            self._client = ChatOpenAI(
                api_key=self.config.api_key,
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._client = ChatAnthropic(
                api_key=self.config.api_key,
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                self._client = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.base_url,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens,
                )
            except ImportError:
                raise RuntimeError(
                    "langchain-ollama 未安装。运行: pip install langchain-ollama"
                )
        else:
            raise ValueError(f"不支持的 provider: {self.config.provider}")

        logger.info("LLM 客户端初始化 | provider=%s | model=%s",
                     self.config.provider, self.config.model_name)
        return self._client

    async def ainvoke(self, messages: List[Any]) -> dict:
        """异步调用 LLM，返回标准格式 result。"""
        client = self._get_client()
        response = await client.ainvoke(messages)
        return {"messages": [response]}

    def invoke(self, messages: List[Any]) -> dict:
        """同步调用 LLM。"""
        client = self._get_client()
        response = client.invoke(messages)
        return {"messages": [response]}
