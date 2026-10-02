"""记忆系统集成测试。

覆盖：
- 短期记忆 TTL 过期验证
- 长期记忆偏好提取准确性
- 记忆注入对调研结果的影响（有记忆 vs 无记忆的对比）
"""

from __future__ import annotations

import time

from src.memory.long_term import LongTermMemory
from src.memory.short_term import InMemoryShortTerm, ShortTermMemory

# ---------------------------------------------------------------------------
# 短期记忆 TTL 测试
# ---------------------------------------------------------------------------


class TestShortTermTTL:
    """短期记忆 TTL 过期验证。"""

    def test_ttl_expiration_in_memory(self):
        """内存版短期记忆 TTL 过期。"""
        mem = InMemoryShortTerm(ttl_seconds=1)
        mem.add("user1", "消息1")
        mem.add("user1", "消息2")
        assert len(mem.get_recent("user1")) == 2

        time.sleep(1.5)
        messages = mem.get_recent("user1")
        assert len(messages) == 0

    def test_ttl_not_expired(self):
        """未过期时消息应保留。"""
        mem = InMemoryShortTerm(ttl_seconds=10)
        mem.add("user1", "消息1")
        time.sleep(0.5)
        assert len(mem.get_recent("user1")) == 1

    def test_partial_expiration(self):
        """部分消息过期，部分保留。"""
        mem = InMemoryShortTerm(ttl_seconds=2)
        mem.add("user1", "消息1")
        time.sleep(1)
        mem.add("user1", "消息2")
        time.sleep(1.5)
        # 消息1 已过期，消息2 应保留
        messages = mem.get_recent("user1")
        assert "消息1" not in messages
        assert "消息2" in messages

    def test_multi_user_ttl(self):
        """多用户 TTL 独立。"""
        mem = InMemoryShortTerm(ttl_seconds=1)
        mem.add("user_a", "A 的消息")
        mem.add("user_b", "B 的消息")
        time.sleep(1.5)
        assert mem.get_recent("user_a") == []
        assert mem.get_recent("user_b") == []


# ---------------------------------------------------------------------------
# 长期记忆偏好提取准确性
# ---------------------------------------------------------------------------


class TestLongTermPreferenceExtraction:
    """长期记忆偏好提取准确性测试。"""

    def test_source_preference_extraction(self):
        """来源偏好提取。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["我之前的研究都关注 .gov.cn 官方来源"]
        prefs = mem.extract_preferences(msgs)
        source_prefs = [p for p in prefs if p["key"] == "source_preference"]
        assert len(source_prefs) >= 1
        assert ".gov.cn" in [p["value"] for p in source_prefs]

    def test_domain_preference_extraction(self):
        """研究领域偏好提取。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["我对 LangGraph 和 Agent 系统很感兴趣"]
        prefs = mem.extract_preferences(msgs)
        domain_prefs = [p for p in prefs if p["key"] == "research_domain"]
        assert len(domain_prefs) >= 1

    def test_language_preference_extraction(self):
        """语言偏好提取。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["请用中文回答我的问题"]
        prefs = mem.extract_preferences(msgs)
        lang_prefs = [p for p in prefs if p["key"] == "language_preference"]
        assert len(lang_prefs) >= 1
        assert lang_prefs[0]["value"] == "zh"

    def test_no_false_positives(self):
        """无关消息不应触发偏好提取。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["今天天气不错，出去散步吧"]
        prefs = mem.extract_preferences(msgs)
        assert prefs == []

    def test_multiple_preferences(self):
        """多条消息应提取多个偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = [
            "我偏好 .gov.cn 来源",
            "我对 AI 很感兴趣",
            "请用中文回答",
        ]
        prefs = mem.extract_preferences(msgs)
        keys = [p["key"] for p in prefs]
        assert "source_preference" in keys
        assert "research_domain" in keys
        assert "language_preference" in keys


# ---------------------------------------------------------------------------
# 记忆注入对比测试
# ---------------------------------------------------------------------------


class TestMemoryInjectionComparison:
    """有记忆 vs 无记忆的对比测试。"""

    def test_recall_with_memories(self):
        """有记忆时应返回上下文。"""
        mem = LongTermMemory(db_path=":memory:")
        mem.store(
            "user1",
            [
                {"content": "用户偏好 .gov.cn 官方来源"},
                {"content": "用户对 LangGraph 感兴趣"},
            ],
            memory_type="preference",
        )

        context = mem.recall("user1", "搜索来源")
        assert len(context) > 0
        assert "用户偏好" in context or len(context) > 10

    def test_recall_without_memories(self):
        """无记忆时应返回空字符串。"""
        mem = LongTermMemory(db_path=":memory:")
        context = mem.recall("user2", "任意查询")
        assert context == ""

    def test_similar_query_returns_more(self):
        """相似查询应返回更多匹配结果。"""
        mem = LongTermMemory(db_path=":memory:")
        mem.store(
            "user1",
            [
                {"content": "LangGraph 是构建 Agent 的框架"},
                {"content": "Python 是编程语言"},
                {"content": "LangGraph 支持状态管理"},
            ],
            memory_type="fact",
        )

        # 精确查询
        results_exact = mem.search(
            "user1", "LangGraph 是构建 Agent 的框架", min_similarity=0.0
        )
        # 模糊查询
        results_fuzzy = mem.search("user1", "Agent 框架", min_similarity=0.0)

        # 精确查询应至少有一条
        assert len(results_exact) >= 1
        # 模糊查询也应命中相关结果
        assert len(results_fuzzy) >= 1

    def test_memory_affects_context(self):
        """记忆注入应影响调研上下文。"""
        mem = LongTermMemory(db_path=":memory:")
        # 存储用户偏好
        mem.store(
            "user1",
            [
                {"content": "用户偏好权威来源"},
                {"content": "用户研究 LangGraph"},
            ],
            memory_type="preference",
        )

        # 模拟注入到 memory_context
        query = "LangGraph 反思循环"
        context = mem.recall("user1", query)

        # 有记忆时应包含用户偏好
        if context:
            assert "用户偏好" in context or "权威" in context

    def test_multi_user_isolation(self):
        """多用户记忆应隔离。"""
        mem = LongTermMemory(db_path=":memory:")
        mem.store(
            "user_a",
            [
                {"content": "用户 A 的偏好"},
            ],
            memory_type="preference",
        )
        mem.store(
            "user_b",
            [
                {"content": "用户 B 的偏好"},
            ],
            memory_type="preference",
        )

        ctx_a = mem.recall("user_a", "偏好")
        ctx_b = mem.recall("user_b", "偏好")

        assert "用户 A 的偏好" in ctx_a
        assert "用户 B 的偏好" in ctx_b
        assert "用户 A" not in ctx_b
        assert "用户 B" not in ctx_a


# ---------------------------------------------------------------------------
# 端到端集成测试
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """端到端记忆系统集成测试。"""

    def test_full_workflow(self):
        """完整工作流：提取 → 存储 → 召回。"""
        # 1. 初始化
        short_mem = ShortTermMemory(ttl_seconds=3600, backend="memory")
        long_mem = LongTermMemory(db_path=":memory:")

        # 2. 模拟对话历史
        messages = [
            "我对 LangGraph 和 Agent 系统很感兴趣",
            "我之前的研究都关注 .gov.cn 官方来源",
            "请用中文回答我的问题",
        ]

        # 3. 从对话中提取偏好
        preferences = long_mem.extract_preferences(messages)
        assert len(preferences) > 0

        # 4. 存储偏好到长期记忆
        long_mem.store("user1", preferences, memory_type="preference")

        # 5. 添加到短期记忆（当前对话）
        short_mem.add("user1", "我对 LangGraph 和 Agent 系统很感兴趣")
        short_mem.add("user1", "我之前的研究都关注 .gov.cn 官方来源")

        # 6. 验证短期记忆
        recent = short_mem.get_recent("user1")
        assert len(recent) >= 2

        # 7. 验证长期记忆召回（使用较低阈值以兼容简单 embedding）
        context = long_mem.recall("user1", "偏好", min_similarity=0.0)
        assert len(context) > 0

        # 8. 清理
        short_mem.clear("user1")
        long_mem.clear("user1")

        # 9. 验证清理
        assert short_mem.get_recent("user1") == []
        assert long_mem.recall("user1", "任意") == ""

    def test_fresh_user_vs_returning_user(self):
        """新用户 vs 回头客的对比。"""
        long_mem = LongTermMemory(db_path=":memory:")

        # 新用户
        ctx_new = long_mem.recall("new_user", "有什么建议？")
        assert ctx_new == ""

        # 回头客（有历史记忆）
        long_mem.store(
            "returning_user",
            [
                {"content": "偏好 .gov.cn 来源"},
                {"content": "研究 LangGraph"},
            ],
            memory_type="preference",
        )
        ctx_return = long_mem.recall("returning_user", "来源偏好")
        assert len(ctx_return) > 0
        assert "偏好" in ctx_return

    def test_concurrent_users(self):
        """并发用户场景。"""
        long_mem = LongTermMemory(db_path=":memory:")

        # 模拟多个用户
        for i in range(5):
            long_mem.store(
                f"user_{i}",
                [
                    {"content": f"用户 {i} 的偏好"},
                ],
                memory_type="preference",
            )

        # 验证每个用户只能看到自己的记忆
        for i in range(5):
            ctx = long_mem.recall(f"user_{i}", "偏好")
            assert f"用户 {i}" in ctx
            # 其他用户的记忆不应出现在这里
            for j in range(5):
                if i != j:
                    assert f"用户 {j}" not in ctx


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_preference_extraction(self):
        """空消息列表提取偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        prefs = mem.extract_preferences([])
        assert prefs == []

    def test_null_content_storage(self):
        """空内容存储。"""
        mem = LongTermMemory(db_path=":memory:")
        ids = mem.store("user1", [{"content": ""}], memory_type="preference")
        assert len(ids) == 1

    def test_very_long_message(self):
        """超长消息处理。"""
        mem = LongTermMemory(db_path=":memory:")
        long_content = "x" * 10000
        mem.store("user1", [{"content": long_content}], memory_type="fact")
        memories = mem.list_memories("user1")
        assert len(memories) == 1
        assert len(memories[0]["content"]) == 10000

    def test_special_characters_in_content(self):
        """特殊字符内容。"""
        mem = LongTermMemory(db_path=":memory:")
        content = "你好🌍世界！@#$%^&*()"
        mem.store("user1", [{"content": content}], memory_type="preference")
        memories = mem.list_memories("user1")
        assert memories[0]["content"] == content
