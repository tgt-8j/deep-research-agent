"""工作流取消机制：支持用户中途停止运行。

参考 Paper-Agent 的 WorkflowCancellation 模式，实现轻量级的
取消信号机制，节点在关键步骤检查是否收到停止请求。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class WorkflowCancellation:
    """保存一次运行是否收到用户停止请求。

    用法：
        cancellation = WorkflowCancellation()
        # 用户点击"停止"时调用
        cancellation.request()
        # 节点执行中定期检查
        cancellation.raise_if_requested()  # 如果已请求则抛出 CancelledError
    """

    def __init__(self) -> None:
        self._requested = False
        self._event = asyncio.Event()

    def request(self) -> None:
        """记录用户已经请求停止，并唤醒正在等待这个信号的代码。"""
        self._requested = True
        self._event.set()

    def is_requested(self) -> bool:
        """返回用户是否已经请求停止。"""
        return self._requested

    async def wait(self) -> None:
        """等待用户发出停止请求，供需要主动等待的任务使用。"""
        await self._event.wait()

    def raise_if_requested(self) -> None:
        """在安全边界检查停止请求，收到请求后让当前异步任务退出。"""
        if self._requested:
            raise asyncio.CancelledError()
