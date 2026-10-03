"""Redis 短期记忆测试

测试 RedisShortTermMemory 的基本操作、压缩、失败降级。
需要运行中的 Redis 实例（默认 redis://127.0.0.1:6379/0）。
设置 REDIS_TEST_URL 环境变量可指定其他地址。

运行: pytest tests/test_redis_short_term.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 确保 app 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

REDIS_TEST_URL = os.getenv("REDIS_TEST_URL", "redis://127.0.0.1:6379/0")


def _get_redis():
    """获取 Redis 客户端，连接失败返回 None"""
    try:
        import redis

        r = redis.Redis.from_url(REDIS_TEST_URL, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


@pytest.fixture(scope="module")
def redis_client():
    """模块级 Redis 客户端 fixture"""
    return _get_redis()


@pytest.fixture(scope="module", autouse=True)
def clean_redis_keys(redis_client):
    """清理测试 key，确保每次测试干净环境"""
    if not redis_client:
        yield None
        return
    # 先清理
    keys = redis_client.keys("ma:short:*")
    if keys:
        redis_client.delete(*keys)
    yield None
    # 清理
    keys = redis_client.keys("ma:short:*")
    if keys:
        redis_client.delete(*keys)


@pytest.mark.skipif(not _get_redis(), reason="Redis 不可用")
class TestRedisShortTermMemory:
    """集成测试：需要真实 Redis"""

    def test_connection_and_ping(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        assert mem.is_alive()
        assert mem._client is not None

    def test_save_and_get_messages(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        tid = "test_thread_save"
        mem.add_message(tid, HumanMessage(content="你好"), user_id="u1", tenant_id="t1")
        msgs = mem.get_messages(
            tid, include_summary=False, user_id="u1", tenant_id="t1"
        )
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "你好"

    def test_save_multiple_messages(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        tid = "test_thread_multi"
        for i in range(5):
            mem.add_message(
                tid, HumanMessage(content=f"消息{i}"), user_id="u1", tenant_id="t1"
            )
            mem.add_message(
                tid, AIMessage(content=f"AI回复{i}"), user_id="u1", tenant_id="t1"
            )
        msgs = mem.get_messages(
            tid, include_summary=False, user_id="u1", tenant_id="t1"
        )
        assert len(msgs) == 10

    def test_summary_storage(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(
            redis_url=REDIS_TEST_URL,
            max_messages=3,
            summary_threshold=2,
        )
        tid = "test_thread_summary"
        for i in range(5):
            mem.add_message(
                tid, HumanMessage(content=f"消息{i}"), user_id="u1", tenant_id="t1"
            )
        # 压缩后保留 summary_threshold 条最近消息，旧消息转为摘要
        msgs = mem.get_messages(
            tid, include_summary=False, user_id="u1", tenant_id="t1"
        )
        assert len(msgs) <= 3  # 不超过 max_messages
        summary = mem.get_summary(tid, user_id="u1", tenant_id="t1")
        assert summary  # 应有摘要

    def test_list_active_threads(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        mem.add_message(
            "thread_a", HumanMessage(content="a"), user_id="u1", tenant_id="t1"
        )
        mem.add_message(
            "thread_b", HumanMessage(content="b"), user_id="u1", tenant_id="t1"
        )
        threads = mem.list_namespaces()
        assert "thread_a" in threads
        assert "thread_b" in threads

    def test_clear_by_thread(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        mem.add_message(
            "thread_clear", HumanMessage(content="x"), user_id="u1", tenant_id="t1"
        )
        deleted = mem.clear(user_id=None, namespace="thread_clear")
        assert deleted >= 1
        msgs = mem.get_messages("thread_clear", user_id="u1", tenant_id="t1")
        assert len(msgs) == 0

    def test_ttl_expiration(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(
            redis_url=REDIS_TEST_URL,
            ttl_seconds=1,  # 1秒过期
        )
        mem.add_message(
            "thread_ttl", HumanMessage(content="test"), user_id="u1", tenant_id="t1"
        )
        import time

        time.sleep(1.5)
        # TTL 过期后键应该不存在
        r = mem._client
        assert r is not None
        key = mem._thread_key("t1", "u1", "thread_ttl")
        assert r.exists(key) == 0


class TestRedisShortTermMemoryMock:
    """单元测试：使用 mock，不依赖真实 Redis"""

    def test_failover_when_redis_down(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        # 模拟 Redis 不可用
        with patch("mult_agents.memory.short_term.redis") as mock_redis:
            mock_redis.Redis.from_url.side_effect = Exception("Connection refused")
            mem = RedisShortTermMemory(redis_url="redis://127.0.0.1:6379/0")
            assert mem._client is None
            # 应该降级到内存
            mem.add_message(
                "t1", HumanMessage(content="hi"), user_id="u1", tenant_id="t1"
            )
            msgs = mem.get_messages("t1", user_id="u1", tenant_id="t1")
            assert len(msgs) == 1
            assert msgs[0].content == "hi"

    def test_is_alive_false_when_disconnected(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url="redis://127.0.0.1:9999")
        assert not mem.is_alive()

    def test_serialize_deserialize(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)

        h_msg = HumanMessage(content="hello")
        a_msg = AIMessage(content="reply")
        s_msg = SystemMessage(content="system")

        assert mem._serialize(h_msg) == {"role": "human", "content": "hello"}
        assert mem._serialize(a_msg) == {"role": "ai", "content": "reply"}
        assert mem._serialize(s_msg) == {"role": "system", "content": "system"}

        assert isinstance(
            mem._deserialize({"role": "human", "content": "x"}), HumanMessage
        )
        assert isinstance(mem._deserialize({"role": "ai", "content": "y"}), AIMessage)
        assert isinstance(
            mem._deserialize({"role": "system", "content": "z"}), SystemMessage
        )

    def test_key_pattern(self):
        from mult_agents.memory.short_term import RedisShortTermMemory

        mem = RedisShortTermMemory(redis_url=REDIS_TEST_URL)
        assert mem._thread_key("t1", "u1", "thread1") == "ma:short:t1:u1:thread1"
        assert (
            mem._summary_key("t1", "u1", "thread1") == "ma:short:summary:t1:u1:thread1"
        )
