"""config.py AppConfig 测试"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.config import AppConfig


class TestResolveHelpers:
    def test_resolve_str_from_env(self):
        with patch.dict(os.environ, {"TEST_KEY": "env_value"}):
            result = AppConfig._resolve_str({}, "field", "TEST_KEY", "default")
            assert result == "env_value"

    def test_resolve_str_from_dict(self):
        result = AppConfig._resolve_str({"field": "dict_value"}, "field", "MISSING_KEY", "default")
        assert result == "dict_value"

    def test_resolve_str_fallback_to_default(self):
        result = AppConfig._resolve_str({}, "field", "MISSING_KEY", "default")
        assert result == "default"

    def test_resolve_int(self):
        result = AppConfig._resolve_int({}, "field", "TEST_INT", "42")
        assert result == 42

    def test_resolve_bool_true(self):
        result = AppConfig._resolve_bool({}, "f", "TEST_BOOL", "true")
        assert result is True

    def test_resolve_bool_false(self):
        with patch.dict(os.environ, {"TEST_BOOL": "false"}, clear=True):
            result = AppConfig._resolve_bool({}, "f", "TEST_BOOL", "false")
            assert result is False

    def test_env_overrides_dict(self):
        with patch.dict(os.environ, {"TEST_FIELD": "from_env"}):
            result = AppConfig._resolve_str({"field": "from_dict"}, "field", "TEST_FIELD", "default")
            assert result == "from_env"


class TestWithOverrides:
    def test_override_fields(self):
        config = AppConfig(
            api_key="k", model="m", thread_id="t1", user_id="u1",
            tenant_id="te1", max_iterations=3, enable_memory=True,
            short_term_ttl_seconds=3600, short_term_max_messages=20,
            short_term_summary_threshold=10, short_term_backend="memory",
            long_term_backend="memory", long_term_scope="user",
            save_conversation_task=False, checkpointer_backend="memory",
            enable_milvus=False, memory_top_k=5, redis_url="",
            postgres_dsn="", milvus_host="", milvus_port=19530,
            milvus_collection="c", redis_failover_enabled=True,
        )
        new = config.with_overrides(model="qwen-max", max_iterations=10)
        assert new.model == "qwen-max"
        assert new.max_iterations == 10
        assert new.thread_id == "t1"

    def test_none_values_ignored(self):
        config = AppConfig(
            api_key="k", model="m", thread_id="t1", user_id="u1",
            tenant_id="te1", max_iterations=3, enable_memory=True,
            short_term_ttl_seconds=3600, short_term_max_messages=20,
            short_term_summary_threshold=10, short_term_backend="memory",
            long_term_backend="memory", long_term_scope="user",
            save_conversation_task=False, checkpointer_backend="memory",
            enable_milvus=False, memory_top_k=5, redis_url="",
            postgres_dsn="", milvus_host="", milvus_port=19530,
            milvus_collection="c", redis_failover_enabled=True,
        )
        new = config.with_overrides(model=None)
        assert new.model == "m"


class TestAppConfigDefaults:
    def test_dataclass_has_redis_failover_field(self):
        # Verify the field exists with default value
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(AppConfig)}
        assert "redis_failover_enabled" in fields
        assert fields["redis_failover_enabled"].default is True
