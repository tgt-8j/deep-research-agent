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
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("backend.task_queue")
_TASKS_DIR = Path(__file__).resolve().parents[3] / ".task_queue_data"


@dataclass
class TaskResult:
    task_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    result: Any = None
    error: str = ""


class TaskQueue:
    def __init__(self, max_concurrent: int = 5, persist_path: Path | None = None):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: dict[str, TaskResult] = {}
        self._persist_path = (
            persist_path if persist_path is not None else _TASKS_DIR / "tasks.json"
        )
        self._load_persisted()

    def _load_persisted(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            with self._persist_path.open("r", encoding="utf-8") as f:
                items = json.load(f)
            for tid, item in items.items():
                if item["status"] in {"completed", "failed"}:
                    self._results[tid] = TaskResult(**item)
            logger.info("restored %d persisted tasks", len(self._results))
        except Exception as exc:
            logger.warning("failed to load task persistence: %s", exc)

    def _save_persisted(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = {
                tid: {
                    "task_id": r.task_id,
                    "status": r.status,
                    "result": r.result,
                    "error": r.error,
                }
                for tid, r in self._results.items()
                if r.status in {"completed", "failed"}
            }
            with self._persist_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("failed to save task persistence: %s", exc)

    async def submit(self, task_id: str, coro_factory: Callable[[], Any]) -> TaskResult:
        """提交异步任务。

        Args:
            task_id: 任务唯一标识
            coro_factory: 返回 coroutine 的工厂函数（延迟创建）
        """
        if task_id in self._results:
            logger.info("task %s already exists, returning existing result", task_id)
            return self._results[task_id]

        self._results[task_id] = TaskResult(task_id=task_id, status="pending")

        async def _run():
            self._results[task_id] = TaskResult(task_id=task_id, status="running")
            try:
                result = await coro_factory()
                self._results[task_id] = TaskResult(
                    task_id=task_id, status="completed", result=result
                )
                self._save_persisted()
            except Exception as exc:
                logger.exception("任务 %s 执行失败", task_id)
                self._results[task_id] = TaskResult(
                    task_id=task_id, status="failed", error=str(exc)
                )
                self._save_persisted()

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
            tid
            for tid, r in self._results.items()
            if r.status in {"pending", "running"}
        ]


# 模块级单例（与 WorkflowService 同生命周期）
task_queue = TaskQueue(max_concurrent=10)
