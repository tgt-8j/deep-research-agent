"""记忆系统迁移层：将旧的 MemoryManager 适配到新架构。

从 app/mult_agents/memory/manager.py 迁移，保持向后兼容。
新架构中通过 ResearchRuntimeContext 注入 MemoryManager 实例。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接从旧模块导入，保持兼容性
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from app.mult_agents.memory.manager import MemoryManager
    from app.mult_agents.memory.base import MemoryType, MemoryEntry
    from app.mult_agents.memory.utils import (
        extract_memory_from_messages,
        format_memories_for_prompt,
        create_memory_checkpoint,
    )
except ImportError as e:
    MemoryManager = None  # type: ignore
    MemoryType = None  # type: ignore
    MemoryEntry = None  # type: ignore
    extract_memory_from_messages = None  # type: ignore
    format_memories_for_prompt = None  # type: ignore
    create_memory_checkpoint = None  # type: ignore
    print(f"Warning: memory import failed: {e}", file=sys.stderr)

__all__ = [
    "MemoryManager",
    "MemoryType",
    "MemoryEntry",
    "extract_memory_from_messages",
    "format_memories_for_prompt",
    "create_memory_checkpoint",
]
