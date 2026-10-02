"""LLM Adapter 多 Provider 测试。"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestModelConfig:
    """ModelConfig 字段测试。"""

    def test_dashscope_default(self):
        from src.llm.adapters import ModelConfig
        cfg = ModelConfig(provider="dashscope", model_name="qwen-plus", api_key="k", base_url="https://test")
        assert cfg.provider == "dashscope"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 4096

    def test_openai_config(self):
        from src.llm.adapters import ModelConfig
        cfg = ModelConfig(provider="openai", model_name="gpt-4o", api_key="k", base_url="https://api.openai.com", temperature=0.7, max_tokens=2048)
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_anthropic_config(self):
        from src.llm.adapters import ModelConfig
        cfg = ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet", api_key="k", base_url="https://api.anthropic.com")
        assert cfg.provider == "anthropic"

    def test_ollama_config(self):
        from src.llm.adapters import ModelConfig
        cfg = ModelConfig(provider="ollama", model_name="llama3.2", api_key="", base_url="http://localhost:11434")
        assert cfg.provider == "ollama"

    def test_invalid_provider_raises(self):
        from src.llm.adapters import ModelConfig, LLMAdapter
        cfg = ModelConfig(provider="unknown", model_name="x", api_key="k", base_url="http://x")
        adapter = LLMAdapter(cfg)
        with pytest.raises(ValueError, match="不支持的 provider"):
            adapter._get_client()


class TestLLMAdapterProviders:
    """LLMAdapter provider dispatch 测试。"""

    def test_dashscope_uses_chatopenai(self):
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="dashscope", model_name="qwen-plus", api_key="test-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        adapter = LLMAdapter(cfg)

        mock_client = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=MagicMock(return_value=mock_client))}):
            client = adapter._get_client()
            assert client is mock_client

    def test_anthropic_uses_chatanthropic(self):
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet", api_key="test-key", base_url="https://api.anthropic.com")
        adapter = LLMAdapter(cfg)

        mock_client = MagicMock()
        mock_mod = MagicMock()
        mock_mod.ChatAnthropic = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"langchain_anthropic": mock_mod}):
            client = adapter._get_client()
            assert client is mock_client

    def test_ollama_requires_langchain_ollama(self):
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="ollama", model_name="llama3.2", api_key="", base_url="http://localhost:11434")
        adapter = LLMAdapter(cfg)

        # langchain_ollama 未安装时应抛出 RuntimeError
        with patch.dict("sys.modules", {"langchain_ollama": None}):
            with pytest.raises(RuntimeError, match="langchain-ollama 未安装"):
                adapter._get_client()

    def test_openai_and_dashscope_same_path(self):
        """openai 和 dashscope 都走 ChatOpenAI（兼容端点）。"""
        from src.llm.adapters import LLMAdapter, ModelConfig
        for provider in ("openai", "dashscope"):
            cfg = ModelConfig(provider=provider, model_name="test", api_key="k", base_url="https://test")
            adapter = LLMAdapter(cfg)
            mock_client = MagicMock()
            mock_mod = MagicMock()
            mock_mod.ChatOpenAI = MagicMock(return_value=mock_client)
            with patch.dict("sys.modules", {"langchain_openai": mock_mod}):
                client = adapter._get_client()
                assert client is mock_client

    def test_invoke_delegates_to_client(self):
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="dashscope", model_name="qwen-plus", api_key="k", base_url="https://test")
        adapter = LLMAdapter(cfg)

        mock_client = MagicMock()
        mock_response = MagicMock(content="test content", type="ai")
        mock_client.invoke.return_value = mock_response
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=MagicMock(return_value=mock_client))}):
            result = adapter.invoke([])
            assert result["messages"] == [mock_response]

    def test_ainvoke_delegates_to_client(self):
        import asyncio
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="dashscope", model_name="qwen-plus", api_key="k", base_url="https://test")
        adapter = LLMAdapter(cfg)

        mock_client = MagicMock()
        mock_response = MagicMock(content="async content", type="ai")
        async def mock_ainvoke(_):
            return mock_response
        mock_client.ainvoke = mock_ainvoke

        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=MagicMock(return_value=mock_client))}):
            result = asyncio.run(adapter.ainvoke([]))
            assert result["messages"] == [mock_response]

    def test_cached_client(self):
        """客户端应被缓存，第二次调用不重新创建。"""
        from src.llm.adapters import LLMAdapter, ModelConfig
        cfg = ModelConfig(provider="dashscope", model_name="qwen-plus", api_key="k", base_url="https://test")
        adapter = LLMAdapter(cfg)

        mock_client = MagicMock()

        def factory(*args, **kwargs):
            return mock_client

        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=MagicMock(side_effect=factory))}):
            c1 = adapter._get_client()
            c2 = adapter._get_client()
            assert c1 is c2 is mock_client


class TestFactory:
    """LLM factory 测试。"""

    def test_create_default_llm_uses_env(self):
        import os
        from src.llm.factory import create_default_llm
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "env-key",
            "MODEL": "qwen-max",
            "BASE_URL": "https://custom.url",
        }):
            adapter = create_default_llm()
            assert adapter.config.model_name == "qwen-max"
            assert adapter.config.api_key == "env-key"
            assert adapter.config.base_url == "https://custom.url"

    def test_create_llm_for_agent(self):
        from src.llm.factory import create_llm_for_agent
        adapter = create_llm_for_agent(model="gpt-4o", api_key="my-key", temperature=0.7)
        assert adapter.config.model_name == "gpt-4o"
        assert adapter.config.api_key == "my-key"
        assert adapter.config.temperature == 0.7
