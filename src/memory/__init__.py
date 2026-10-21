"""记忆系统：短期记忆（内存 / Redis）+ 长期记忆（SQLite + Embedding）。

历史沿革：本模块原先是一层「legacy MemoryManager 适配壳」，会在导入时
去 import ``app.mult_agents.memory``，导致新架构反向依赖旧包。现已移除：
``src/`` 不再引用 ``app``，需要旧实现请直接 import 旧包。

用法（按需导入子模块，避免一次性把 redis/sqlite 依赖拉起来）::

    from src.memory.short_term import ShortTermMemory, InMemoryShortTerm
    from src.memory.long_term import LongTermMemory, create_long_term_memory
"""

from __future__ import annotations

__all__: list[str] = []
