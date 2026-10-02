"""长期记忆模块测试。

验证 LongTermMemory 的存储、召回、偏好提取、语义搜索等功能。
"""

from __future__ import annotations

import pytest

from src.memory.long_term import LongTermMemory, create_long_term_memory

# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------


@pytest.fixture
def long_mem():
    """创建内存数据库的 LongTermMemory 实例。"""
    return LongTermMemory(db_path=":memory:")


# ---------------------------------------------------------------------------
# 存储与召回
# ---------------------------------------------------------------------------


class TestStoreAndRecall:
    def test_store_and_recall(self, long_mem):
        """存储并召回记忆。"""
        long_mem.store(
            "user1",
            [
                {"content": "用户偏好 .gov.cn 官方来源"},
                {"content": "用户对 AI Agent 领域感兴趣"},
            ],
            memory_type="preference",
        )

        context = long_mem.recall("user1", "搜索来源偏好")
        assert "用户偏好" in context or len(context) > 0

    def test_recall_empty_user(self, long_mem):
        """不存在的用户应返回空字符串。"""
        context = long_mem.recall("unknown_user", "任意查询")
        assert context == ""

    def test_recall_empty_query(self, long_mem):
        """空查询应返回空结果。"""
        context = long_mem.recall("user1", "")
        assert context == ""


# ---------------------------------------------------------------------------
# 偏好提取
# ---------------------------------------------------------------------------


class TestExtractPreferences:
    def test_extract_source_preference(self):
        """提取来源偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["我之前的研究都关注 .gov.cn 官方来源"]
        prefs = mem.extract_preferences(msgs)
        assert any(p["key"] == "source_preference" for p in prefs)

    def test_extract_domain_preference(self):
        """提取研究领域偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["我对 LangGraph 和 Agent 系统很感兴趣"]
        prefs = mem.extract_preferences(msgs)
        assert any(p["key"] == "research_domain" for p in prefs)

    def test_extract_language_preference(self):
        """提取语言偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        msgs = ["请用中文回答我的问题"]
        prefs = mem.extract_preferences(msgs)
        assert any(p["key"] == "language_preference" for p in prefs)

    def test_empty_messages(self):
        """空消息列表应返回空偏好。"""
        mem = LongTermMemory(db_path=":memory:")
        prefs = mem.extract_preferences([])
        assert prefs == []

    def test_no_matching_patterns(self):
        """无匹配模式时返回空列表。"""
        mem = LongTermMemory(db_path=":memory:")
        prefs = mem.extract_preferences(["这是一条无关消息"])
        assert prefs == []


# ---------------------------------------------------------------------------
# 用户档案
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_get_empty_profile(self, long_mem):
        """新用户的档案应为空字典。"""
        profile = long_mem.get_profile("new_user")
        assert profile == {}

    def test_update_and_get_profile(self, long_mem):
        """更新并获取用户档案。"""
        profile = {"name": "Alice", "preferences": {"source": ".gov.cn"}}
        long_mem.update_profile("user1", profile)
        retrieved = long_mem.get_profile("user1")
        assert retrieved["name"] == "Alice"
        assert retrieved["preferences"]["source"] == ".gov.cn"

    def test_update_overwrites(self, long_mem):
        """更新档案应覆盖旧值。"""
        long_mem.update_profile("user1", {"age": 25})
        long_mem.update_profile("user1", {"age": 30, "name": "Bob"})
        retrieved = long_mem.get_profile("user1")
        assert retrieved["age"] == 30
        assert retrieved["name"] == "Bob"


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_results(self, long_mem):
        """搜索应返回匹配结果。"""
        long_mem.store(
            "user1",
            [
                {"content": "LangGraph 是构建 Agent 的框架"},
                {"content": "Python 是一种编程语言"},
            ],
            memory_type="fact",
        )

        results = long_mem.search("user1", "LangGraph", limit=5)
        assert len(results) >= 1
        assert results[0]["content"] == "LangGraph 是构建 Agent 的框架"

    def test_search_with_type_filter(self, long_mem):
        """按类型过滤搜索。"""
        long_mem.store(
            "user1",
            [
                {"content": "偏好内容", "metadata": {}},
            ],
            memory_type="preference",
        )
        long_mem.store(
            "user1",
            [
                {"content": "事实内容", "metadata": {}},
            ],
            memory_type="fact",
        )

        pref_results = long_mem.search("user1", "内容", memory_type="preference")
        fact_results = long_mem.search("user1", "内容", memory_type="fact")

        assert all(r["memory_type"] == "preference" for r in pref_results)
        assert all(r["memory_type"] == "fact" for r in fact_results)

    def test_search_min_similarity(self, long_mem):
        """最低相似度阈值过滤。"""
        long_mem.store(
            "user1",
            [
                {"content": "完全无关的内容"},
            ],
            memory_type="fact",
        )

        # 高阈值应返回空
        results = long_mem.search("user1", "LangGraph", min_similarity=0.9)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 列出与删除
# ---------------------------------------------------------------------------


class TestListAndDelete:
    def test_list_memories(self, long_mem):
        """列出用户记忆。"""
        long_mem.store(
            "user1",
            [
                {"content": "记忆1"},
                {"content": "记忆2"},
            ],
            memory_type="preference",
        )

        memories = long_mem.list_memories("user1")
        assert len(memories) == 2

    def test_list_with_type_filter(self, long_mem):
        """按类型列出记忆。"""
        long_mem.store(
            "user1",
            [
                {"content": "偏好1"},
            ],
            memory_type="preference",
        )
        long_mem.store(
            "user1",
            [
                {"content": "事实1"},
            ],
            memory_type="fact",
        )

        pref = long_mem.list_memories("user1", memory_type="preference")
        fact = long_mem.list_memories("user1", memory_type="fact")
        assert len(pref) == 1
        assert len(fact) == 1

    def test_delete(self, long_mem):
        """删除记忆。"""
        ids = long_mem.store(
            "user1",
            [
                {"content": "待删除"},
            ],
            memory_type="preference",
        )
        assert len(ids) == 1

        deleted = long_mem.delete(ids[0])
        assert deleted is True
        assert len(long_mem.list_memories("user1")) == 0

    def test_delete_nonexistent(self, long_mem):
        """删除不存在的记忆。"""
        deleted = long_mem.delete("nonexistent_id")
        assert deleted is False

    def test_clear(self, long_mem):
        """清空用户所有记忆。"""
        long_mem.store(
            "user1",
            [
                {"content": "记忆1"},
                {"content": "记忆2"},
            ],
            memory_type="preference",
        )

        count = long_mem.clear("user1")
        assert count == 2
        assert len(long_mem.list_memories("user1")) == 0


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_content(self, long_mem):
        """空内容应正常处理。"""
        long_mem.store(
            "user1",
            [
                {"content": ""},
            ],
            memory_type="preference",
        )
        memories = long_mem.list_memories("user1")
        assert len(memories) == 1

    def test_special_characters(self, long_mem):
        """特殊字符应正常存储。"""
        long_mem.store(
            "user1",
            [
                {"content": "你好🌍世界！@#$%"},
            ],
            memory_type="preference",
        )
        memories = long_mem.list_memories("user1")
        assert memories[0]["content"] == "你好🌍世界！@#$%"

    def test_unicode_content(self, long_mem):
        """Unicode 内容应正常存储。"""
        long_mem.store(
            "user1",
            [
                {"content": "مرحبا بالعالم"},
                {"content": "こんにちは世界"},
            ],
            memory_type="preference",
        )
        memories = long_mem.list_memories("user1")
        assert len(memories) == 2

    def test_multiple_users_isolated(self, long_mem):
        """不同用户的记忆应隔离。"""
        long_mem.store("user_a", [{"content": "A 的记忆"}], memory_type="preference")
        long_mem.store("user_b", [{"content": "B 的记忆"}], memory_type="preference")

        assert len(long_mem.list_memories("user_a")) == 1
        assert len(long_mem.list_memories("user_b")) == 1
        assert long_mem.list_memories("user_a")[0]["content"] == "A 的记忆"
        assert long_mem.list_memories("user_b")[0]["content"] == "B 的记忆"

    def test_factory_function(self):
        """工厂函数应正常工作。"""
        mem = create_long_term_memory(db_path=":memory:")
        assert isinstance(mem, LongTermMemory)
        mem.store("user1", [{"content": "测试"}], memory_type="preference")
        assert len(mem.list_memories("user1")) == 1


# ---------------------------------------------------------------------------
# 相似度计算
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_exact_match(self, long_mem):
        """完全匹配的文档应有最高相似度。"""
        long_mem.store(
            "user1",
            [
                {"content": "LangGraph 是构建 Agent 的框架"},
            ],
            memory_type="fact",
        )

        results = long_mem.search(
            "user1", "LangGraph 是构建 Agent 的框架", min_similarity=0.0
        )
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.5

    def test_different_similarity(self, long_mem):
        """不同内容的相似度应有差异。"""
        long_mem.store(
            "user1",
            [
                {"content": "LangGraph 是构建 Agent 的框架"},
                {"content": "Python 是一种编程语言"},
            ],
            memory_type="fact",
        )

        results = long_mem.search("user1", "LangGraph", min_similarity=0.0)
        assert len(results) == 2
        # LangGraph 相关内容应排名更高
        assert results[0]["content"] == "LangGraph 是构建 Agent 的框架"
