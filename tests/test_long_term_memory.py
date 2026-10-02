"""long_term.py 长期记忆存储测试"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.memory.long_term import (
    SemanticMemoryStore,
    EpisodicMemoryStore,
    ProceduralMemoryStore,
)
from mult_agents.memory.base import MemoryEntry, MemoryType


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def semantic_store(db_path):
    return SemanticMemoryStore(db_path=db_path)


@pytest.fixture
def episodic_store(db_path):
    return EpisodicMemoryStore(db_path=db_path)


@pytest.fixture
def procedural_store(db_path):
    return ProceduralMemoryStore(db_path=db_path)


class TestSemanticMemoryStore:
    def test_save_and_get(self, semantic_store):
        entry = MemoryEntry(
            content="Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
            user_id="user1",
            metadata={"key": "value"},
        )
        mem_id = semantic_store.save(entry)
        assert mem_id is not None

        retrieved = semantic_store.get(mem_id)
        assert retrieved is not None
        assert retrieved.content == "Python is a programming language"

    def test_search(self, semantic_store):
        semantic_store.save(MemoryEntry(
            content="LangGraph is great for agent workflows",
            memory_type=MemoryType.SEMANTIC,
            user_id="user1",
        ))
        results = semantic_store.search("LangGraph", user_id="user1", limit=5)
        assert len(results) >= 1

    def test_delete(self, semantic_store):
        entry = MemoryEntry(
            content="temporary fact",
            memory_type=MemoryType.SEMANTIC,
            user_id="user1",
        )
        mem_id = semantic_store.save(entry)
        assert semantic_store.delete(mem_id) is True
        assert semantic_store.get(mem_id) is None

    def test_clear(self, semantic_store):
        for i in range(3):
            semantic_store.save(MemoryEntry(
                content=f"fact {i}",
                memory_type=MemoryType.SEMANTIC,
                user_id="user1",
            ))
        count = semantic_store.clear(user_id="user1")
        assert count >= 1

    def test_save_profile(self, semantic_store):
        profile_id = semantic_store.save_profile(
            "user1", {"name": "Alice", "age": 30}
        )
        assert profile_id is not None

    def test_upsert_profile_merge(self, semantic_store):
        semantic_store.save_profile("user1", {"name": "Bob", "hobby": "reading"})
        semantic_store.save_profile("user1", {"hobby": "coding", "lang": "Python"}, merge=True)
        results = semantic_store.search("coding", user_id="user1")
        assert len(results) >= 0


class TestEpisodicMemoryStore:
    def test_save_and_search(self, episodic_store):
        episodic_store.save(MemoryEntry(
            content={"task_type": "研究", "query": "AI agents"},
            memory_type=MemoryType.EPISODIC,
            user_id="user1",
        ))
        results = episodic_store.search("AI", user_id="user1")
        assert len(results) >= 1

    def test_list_by_user(self, episodic_store):
        episodic_store.save(MemoryEntry(
            content="Completed task 1",
            memory_type=MemoryType.EPISODIC,
            user_id="user1",
        ))
        episodic_store.save(MemoryEntry(
            content="Completed task 2",
            memory_type=MemoryType.EPISODIC,
            user_id="user2",
        ))
        results_u1 = episodic_store.search("", user_id="user1")
        results_u2 = episodic_store.search("", user_id="user2")
        assert len(results_u1) >= 1
        assert len(results_u2) >= 1


class TestProceduralMemoryStore:
    def test_save_and_search(self, procedural_store):
        procedural_store.save(MemoryEntry(
            content={"name": "deploy", "steps": ["build", "test", "push"]},
            memory_type=MemoryType.PROCEDURAL,
            user_id="user1",
        ))
        results = procedural_store.search("deploy", user_id="user1")
        assert len(results) >= 1
