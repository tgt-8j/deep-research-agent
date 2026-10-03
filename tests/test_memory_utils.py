"""memory/utils.py 工具函数测试"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain_core.messages import HumanMessage
from mult_agents.memory.base import MemoryEntry, MemoryType
from mult_agents.memory.utils import (
    _simple_similarity,
    calculate_memory_relevance,
    compress_memories,
    create_memory_checkpoint,
    extract_memory_from_messages,
    format_memories_for_prompt,
    merge_user_profile,
)


class TestCreateMemoryCheckpoint:
    def test_creates_checkpoint(self):
        state = {"query": "test", "intent": "direct", "plan": None, "analysis": "ok"}
        cp = create_memory_checkpoint("thread_1", state)
        assert "id" in cp
        assert cp["thread_id"] == "thread_1"
        assert cp["state"]["query"] == "test"
        assert "created_at" in cp

    def test_custom_checkpoint_id(self):
        cp = create_memory_checkpoint("t1", {"query": "q"}, checkpoint_id="custom_id")
        assert cp["id"] == "custom_id"


class TestExtractMemoryFromMessages:
    def test_extract_facts(self):
        msgs = [HumanMessage(content="我叫小明，我住在北京。")]
        result = extract_memory_from_messages(
            msgs, extract_facts=True, extract_preferences=False
        )
        assert isinstance(result, dict)
        assert "facts" in result

    def test_extract_preferences(self):
        msgs = [HumanMessage(content="我喜欢Python，偏好简洁的代码风格。")]
        result = extract_memory_from_messages(
            msgs, extract_facts=False, extract_preferences=True
        )
        assert isinstance(result, dict)
        assert "preferences" in result

    def test_empty_messages(self):
        result = extract_memory_from_messages(
            [], extract_facts=True, extract_preferences=True
        )
        assert result["facts"] == []
        assert result["preferences"] == []


class TestFormatMemoriesForPrompt:
    def test_empty_memories(self):
        assert format_memories_for_prompt([]) == ""

    def test_semantic_memories(self):
        entries = [
            MemoryEntry(
                content="Python是一种编程语言",
                memory_type=MemoryType.SEMANTIC,
                user_id="u1",
            )
        ]
        result = format_memories_for_prompt(entries)
        assert "Python" in result

    def test_episodic_memories(self):
        entries = [
            MemoryEntry(
                content={"task_type": "研究", "outcome": "完成了调研"},
                memory_type=MemoryType.EPISODIC,
                user_id="u1",
            )
        ]
        result = format_memories_for_prompt(entries)
        assert "研究" in result or "完成了" in result

    def test_procedural_memories(self):
        entries = [
            MemoryEntry(
                content={"name": "步骤A", "steps": ["第一步", "第二步"]},
                memory_type=MemoryType.PROCEDURAL,
                user_id="u1",
            )
        ]
        result = format_memories_for_prompt(entries)
        assert "步骤A" in result

    def test_max_length_truncation(self):
        entries = [
            MemoryEntry(
                content="X" * 1000,
                memory_type=MemoryType.SEMANTIC,
                user_id="u1",
            )
        ]
        result = format_memories_for_prompt(entries, max_length=100)
        assert len(result) <= 120  # 允许截断标记


class TestMergeUserProfile:
    def test_none_existing(self):
        result = merge_user_profile(None, {"name": "Alice", "age": 30})
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_merge_dicts(self):
        existing = {"name": "Bob", "prefs": {"color": "blue"}}
        new = {"prefs": {"size": "large"}, "age": 25}
        result = merge_user_profile(existing, new)
        assert result["name"] == "Bob"
        assert result["prefs"] == {"color": "blue", "size": "large"}
        assert result["age"] == 25

    def test_merge_lists_dedup(self):
        existing = {"tags": ["python", "ai"]}
        new = {"tags": ["ai", "langgraph"]}
        result = merge_user_profile(existing, new)
        assert set(result["tags"]) == {"python", "ai", "langgraph"}

    def test_updates_timestamp(self):
        result = merge_user_profile({"name": "x"}, {"age": 1})
        assert "_last_updated" in result


class TestCalculateMemoryRelevance:
    def test_exact_match(self):
        entry = MemoryEntry(
            content="Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        score = calculate_memory_relevance("Python", entry)
        assert score > 0

    def test_no_match(self):
        entry = MemoryEntry(
            content="completely unrelated content here",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        score = calculate_memory_relevance("totally different topic", entry)
        assert score >= 0

    def test_time_decay(self):
        old_entry = MemoryEntry(
            content="Python",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now() - timedelta(days=365),
        )
        new_entry = MemoryEntry(
            content="Python",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        old_score = calculate_memory_relevance("Python", old_entry)
        new_score = calculate_memory_relevance("Python", new_entry)
        assert new_score > old_score

    def test_score_capped_at_1(self):
        entry = MemoryEntry(
            content="Python Python Python Python",
            memory_type=MemoryType.SEMANTIC,
            access_count=100,
            created_at=datetime.now(),
        )
        score = calculate_memory_relevance("Python", entry)
        assert score <= 1.0


class TestCompressMemories:
    def test_below_target(self):
        entries = [
            MemoryEntry(content=f"mem{i}", memory_type=MemoryType.SEMANTIC)
            for i in range(5)
        ]
        result = compress_memories(entries, target_count=10)
        assert len(result) == 5

    def test_above_target_dedupe(self):
        entries = [
            MemoryEntry(content="Python programming", memory_type=MemoryType.SEMANTIC)
            for _ in range(20)
        ]
        result = compress_memories(entries, target_count=5)
        assert len(result) <= 5

    def test_preserves_distinct_entries(self):
        entries = [
            MemoryEntry(
                content=f"different content {i}", memory_type=MemoryType.SEMANTIC
            )
            for i in range(15)
        ]
        result = compress_memories(entries, target_count=5)
        assert len(result) == 5


class TestSimpleSimilarity:
    def test_identical(self):
        assert _simple_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _simple_similarity("apple banana", "cherry date") == 0.0

    def test_partial_overlap(self):
        sim = _simple_similarity("the quick brown fox", "the quick red dog")
        assert 0 < sim < 1.0

    def test_empty_strings(self):
        assert _simple_similarity("", "anything") == 0.0
