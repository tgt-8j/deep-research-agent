"""轻量级异步任务队列：基于 asyncio.Queue + Semaphore 的并发控制。

设计：
  - submit(task_id, coro) → 提交异步任务，返回 TaskResult
  - get_status(task_id) → 查询任务状态和结果
  - 最大并发数通过 Semaphore 控制，防止线程池耗尽

注意：此队列仅在当前进程内有效，不适合跨进程分发。
      生产环境可替换为 Celery / ARQ / Dramatiq。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("backend.task_queue")


@dataclass
class TaskResult:
    task_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    result: Any = None
    error: str = ""


class TaskQueue:
    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: dict[str, TaskResult] = {}

    async def submit(self, task_id: str, coro_factory: Callable[[], Any]) -> TaskResult:
        """提交异步任务。

        Args:
            task_id: 任务唯一标识
            coro_factory: 返回 coroutine 的工厂函数（延迟创建）
        """
        if task_id in self._results:
            raise ValueError(f"任务 {task_id} 已存在")

        self._results[task_id] = TaskResult(task_id=task_id, status="pending")

        async def _run():
            self._results[task_id] = TaskResult(task_id=task_id, status="running")
            try:
                result = await coro_factory()
                self._results[task_id] = TaskResult(
                    task_id=task_id, status="completed", result=result
                )
            except Exception as exc:
                logger.exception("任务 %s 执行失败", task_id)
                self._results[task_id] = TaskResult(
                    task_id=task_id, status="failed", error=str(exc)
                )

        # 立即用 semaphore 包装
        async def _guarded_run():
            async with self._semaphore:
                await _run()

        asyncio.create_task(_guarded_run())
        return self._results[task_id]

    def get_status(self, task_id: str) -> TaskResult | None:
        """查询任务状态。"""
        return self._results.get(task_id)

    def list_active(self) -> list[str]:
        """返回所有正在执行的任务 ID。"""
        return [
            tid for tid, r in self._results.items()
            if r.status in {"pending", "running"}
        ]


# 模块级单例（与 WorkflowService 同生命周期）
task_queue = TaskQueue(max_concurrent=10)
