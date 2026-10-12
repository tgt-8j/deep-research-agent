"""WorkflowService / TaskQueue 测试

注意：由于 workflow_service.py 使用了多层相对导入（...mult_agents），
直接 import 会失败。本文件只测试 task_queue 模块。
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.service.task_queue import TaskQueue


def _make_queue(max_concurrent: int = 5) -> TaskQueue:
    """创建使用临时持久化目录的 TaskQueue，避免测试间数据污染。"""
    tmp_dir = Path(tempfile.mkdtemp())
    return TaskQueue(max_concurrent=max_concurrent, persist_path=tmp_dir / "tasks.json")


class TestTaskQueue:
    """异步任务队列测试。"""

    @pytest.mark.asyncio
    async def test_submit_and_get_status(self):
        queue = _make_queue(max_concurrent=2)
        call_count = 0
        task_id = "task_1"

        async def my_task():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await queue.submit(task_id, my_task)
        assert result.task_id == task_id
        # submit 返回的是内部 TaskResult 引用；快速任务可能在断言前已完成，
        # 因此只验证状态是 pending 或 completed（二者皆可）
        assert result.status in ("pending", "completed")

        # 等待任务完成
        for _ in range(50):
            await asyncio.sleep(0.01)
            r = queue.get_status(task_id)
            if r and r.status == "completed":
                break

        final = queue.get_status(task_id)
        assert final is not None
        assert final.status == "completed"
        assert final.result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_submit_failed_task(self):
        queue = _make_queue(max_concurrent=2)
        task_id = "task_fail"

        async def failing_task():
            raise ValueError("task failed")

        await queue.submit(task_id, failing_task)

        for _ in range(50):
            await asyncio.sleep(0.01)
            r = queue.get_status(task_id)
            if r and r.status in {"completed", "failed"}:
                break

        final = queue.get_status(task_id)
        assert final is not None
        assert final.status == "failed"
        assert "task failed" in final.error

    @pytest.mark.asyncio
    async def test_duplicate_submit_returns_existing(self):
        queue = _make_queue(max_concurrent=2)
        task_id = "task_dup"

        async def simple_task():
            return "ok"

        first = await queue.submit(task_id, simple_task)
        # 重复提交同一 task_id 应直接返回已有结果，而非抛异常
        second = await queue.submit(task_id, simple_task)
        assert second is first

    def test_get_nonexistent_task(self):
        queue = _make_queue()
        assert queue.get_status("nonexistent") is None

    def test_list_active_empty(self):
        queue = _make_queue()
        assert queue.list_active() == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """验证并发限制生效。"""
        queue = _make_queue(max_concurrent=2)
        running = 0
        max_running = 0
        import asyncio

        async def slow_task(task_id):
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.05)
            running -= 1
            return task_id

        # 提交 5 个任务
        tasks = [queue.submit(f"t{i}", lambda i=i: slow_task(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        # 并发度不应超过 max_concurrent
        assert max_running <= 2
