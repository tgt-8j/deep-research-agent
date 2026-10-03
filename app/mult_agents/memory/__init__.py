"""
Agent 记忆系统模块

提供短期记忆和长期记忆的统一管理，支持：
- 短期记忆：基于 LangGraph Checkpoint 的线程级记忆
- 长期记忆：基于 PostgreSQL + 向量的持久化记忆
- 语义记忆：用户画像、事实知识
- 情景记忆：历史任务、执行轨迹

Usage:
    from mult_agents.memory import MemoryManager
    
    memory = MemoryManager()
    
    # 短期记忆 - 自动通过 State 管理
    state["messages"].append(new_message)
    
    # 长期记忆 - 显式存储
    memory.save_semantic(user_id="user_123", key="preference", value={"theme": "dark"})
    
    # 检索相关记忆
    memories = memory.search_episodic(user_id="user_123", query="之前的分析任务")
"""

from .base import BaseMemory, MemoryEntry, MemoryType
from .long_term import (
    EpisodicMemoryStore,
    LongTermMemory,
    ProceduralMemoryStore,
    SemanticMemoryStore,
)
from .manager import MemoryManager
from .short_term import ConversationBuffer, RedisShortTermMemory, ShortTermMemory
from .utils import create_memory_checkpoint, extract_memory_from_messages

__all__ = [
    "BaseMemory",
    "ConversationBuffer",
    "EpisodicMemoryStore",
    "LongTermMemory",
    "MemoryEntry",
    "MemoryManager",
    "MemoryType",
    "ProceduralMemoryStore",
    "RedisShortTermMemory",
    "SemanticMemoryStore",
    "ShortTermMemory",
    "create_memory_checkpoint",
    "extract_memory_from_messages",
]
