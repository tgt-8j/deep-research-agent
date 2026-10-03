"""WorkflowService / TaskQueue 测试

注意：由于 workflow_service.py 使用了多层相对导入（...mult_agents），
直接 import 会失败。本文件只测试 task_queue 模块。
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.service.task_queue import TaskQueue


class TestTaskQueue:
    """异步任务队列测试。"""

    @pytest.mark.asyncio
    async def test_submit_and_get_status(self):
        queue = TaskQueue(max_concurrent=2)
        call_count = 0

        async def my_task():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await queue.submit("task_1", my_task)
        assert result.task_id == "task_1"
        assert result.status == "pending"  # 立即返回 pending，后台执行

        # 等待任务完成
        for _ in range(50):
            await asyncio.sleep(0.01)
            r = queue.get_status("task_1")
            if r and r.status == "completed":
                break

        final = queue.get_status("task_1")
        assert final is not None
        assert final.status == "completed"
        assert final.result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_submit_failed_task(self):
        queue = TaskQueue(max_concurrent=2)

        async def failing_task():
            raise ValueError("task failed")

        await queue.submit("task_fail", failing_task)

        for _ in range(50):
            await asyncio.sleep(0.01)
            r = queue.get_status("task_fail")
            if r and r.status in {"completed", "failed"}:
                break

        final = queue.get_status("task_fail")
        assert final is not None
        assert final.status == "failed"
        assert "task failed" in final.error

    @pytest.mark.asyncio
    async def test_duplicate_submit_raises(self):
        queue = TaskQueue(max_concurrent=2)

        async def simple_task():
            return "ok"

        await queue.submit("task_dup", simple_task)
        with pytest.raises(ValueError, match="已存在"):
            await queue.submit("task_dup", simple_task)

    def test_get_nonexistent_task(self):
        queue = TaskQueue()
        assert queue.get_status("nonexistent") is None

    def test_list_active_empty(self):
        queue = TaskQueue()
        assert queue.list_active() == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """验证并发限制生效。"""
        queue = TaskQueue(max_concurrent=2)
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
