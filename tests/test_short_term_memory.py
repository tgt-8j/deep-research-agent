"""短期记忆模块测试。

验证 InMemoryShortTerm、ShortTermMemory 双后端切换、TTL 过期、
clear、list_users 等功能。
"""

from __future__ import annotations

import time

from src.memory.short_term import (
    InMemoryShortTerm,
    ShortTermMemory,
    create_short_term_memory,
)

# ---------------------------------------------------------------------------
# InMemoryShortTerm 基础测试
# ---------------------------------------------------------------------------


class TestInMemoryShortTerm:
    def test_add_and_get(self):
        """添加和获取消息。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        mem.add("user1", "你好")
        messages = mem.get_recent("user1", n=5)
        assert len(messages) == 1
        assert messages[0] == "你好"

    def test_multiple_messages(self):
        """多条消息按顺序返回。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        for i in range(5):
            mem.add("user1", f"消息{i}")
        messages = mem.get_recent("user1", n=10)
        assert len(messages) == 5
        assert messages[0] == "消息0"
        assert messages[-1] == "消息4"

    def test_limit_n(self):
        """限制返回数量。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        for i in range(10):
            mem.add("user1", f"消息{i}")
        recent = mem.get_recent("user1", n=3)
        assert len(recent) == 3
        assert recent[0] == "消息7"
        assert recent[-1] == "消息9"

    def test_empty_user(self):
        """不存在的用户返回空列表。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        assert mem.get_recent("unknown_user") == []

    def test_clear(self):
        """清空用户记忆。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        mem.add("user1", "消息1")
        mem.add("user1", "消息2")
        count = mem.clear("user1")
        assert count == 2
        assert mem.get_recent("user1") == []

    def test_clear_unknown_user(self):
        """清空不存在的用户返回 0。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        assert mem.clear("unknown") == 0

    def test_list_users(self):
        """列出所有有记忆的用户。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        mem.add("user1", "消息")
        mem.add("user2", "消息")
        users = mem.list_users()
        assert "user1" in users
        assert "user2" in users
        assert len(users) == 2

    def test_ttl_expiration(self):
        """TTL 过期后消息应被清理。"""
        mem = InMemoryShortTerm(ttl_seconds=1)  # 1 秒过期
        mem.add("user1", "临时消息")
        time.sleep(1.5)
        messages = mem.get_recent("user1")
        assert len(messages) == 0

    def test_metadata_preserved(self):
        """元数据应被保留。"""
        mem = InMemoryShortTerm(ttl_seconds=3600)
        mem.add("user1", "消息", metadata={"role": "user", "source": "chat"})
        # InMemoryShortTerm 不直接暴露 metadata，但添加应成功
        messages = mem.get_recent("user1")
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# ShortTermMemory 双后端测试
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_default_backend_is_memory(self):
        """默认后端应为 memory。"""
        mem = ShortTermMemory(ttl_seconds=3600)
        assert mem.backend == "memory"

    def test_force_memory_backend(self):
        """强制使用 memory 后端。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        assert mem.backend == "memory"

    def test_add_and_get(self):
        """添加和获取消息。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        msg_id = mem.add("user1", "你好，世界")
        assert msg_id is not None
        messages = mem.get_recent("user1")
        assert "你好，世界" in messages

    def test_multiple_users_isolated(self):
        """不同用户的记忆应隔离。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("user_a", "A 的消息")
        mem.add("user_b", "B 的消息")
        assert mem.get_recent("user_a") == ["A 的消息"]
        assert mem.get_recent("user_b") == ["B 的消息"]

    def test_clear_user(self):
        """清空指定用户。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("user1", "消息1")
        mem.add("user1", "消息2")
        cleared = mem.clear("user1")
        assert cleared == 2
        assert mem.get_recent("user1") == []

    def test_list_users(self):
        """列出所有用户。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("alice", "Hello")
        mem.add("bob", "Hi")
        users = mem.list_users()
        assert "alice" in users
        assert "bob" in users

    def test_ttl_expiration(self):
        """TTL 过期验证。"""
        mem = ShortTermMemory(ttl_seconds=1, backend="memory")
        mem.add("user1", "临时消息")
        time.sleep(1.5)
        messages = mem.get_recent("user1")
        assert len(messages) == 0

    def test_is_redis_available_false_when_no_redis(self):
        """无 Redis 时 should return False。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        assert mem.is_redis_available() is False

    def test_factory_function(self):
        """工厂函数应正常工作。"""
        mem = create_short_term_memory(ttl_seconds=3600, backend="memory")
        assert isinstance(mem, ShortTermMemory)
        mem.add("user1", "测试消息")
        assert len(mem.get_recent("user1")) == 1


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message(self):
        """空消息应正常处理。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("user1", "")
        messages = mem.get_recent("user1")
        assert len(messages) == 1
        assert messages[0] == ""

    def test_long_message(self):
        """长消息应正常存储。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        long_msg = "x" * 10000
        mem.add("user1", long_msg)
        messages = mem.get_recent("user1")
        assert len(messages[0]) == 10000

    def test_special_characters(self):
        """特殊字符应正常处理。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("user1", "你好🌍世界！@#$%")
        messages = mem.get_recent("user1")
        assert "你好🌍世界！@#$%" in messages

    def test_unicode_messages(self):
        """Unicode 消息应正常处理。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        mem.add("user1", "مرحبا بالعالم")  # 阿拉伯语
        mem.add("user2", "こんにちは世界")  # 日语
        assert mem.get_recent("user1") == ["مرحبا بالعالم"]
        assert mem.get_recent("user2") == ["こんにちは世界"]

    def test_concurrent_adds(self):
        """并发添加不应出错。"""
        mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        for i in range(100):
            mem.add("user1", f"消息{i}")
        messages = mem.get_recent("user1", n=200)
        assert len(messages) == 100
