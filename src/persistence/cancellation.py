"""工作流取消机制：支持用户中途停止运行。

取消信号是**请求级**的，但图（`StateGraph`）在 `WorkflowService` 里被缓存复用，
所以信号不能挂在图或节点闭包上 —— 否则两个并发的调研任务会互相取消。

这里的做法：用 `ContextVar` 承载"当前请求的取消信号"。
- asyncio 的每个任务持有独立的 context 副本，并发请求天然隔离；
- 不用把 `WorkflowCancellation`（含 `asyncio.Event`，不可序列化）塞进 state，
  也就不会污染 checkpointer 的快照；
- 节点侧统一通过 `current_cancellation()` 取，不需要改节点签名。

用法（服务层）::

    with cancellation_scope(cancellation):
        await graph.ainvoke(state, config)

用法（生产者，例如取消接口或客户端断开）::

    cancellation.request()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

logger = logging.getLogger("research.cancellation")


class WorkflowCancellation:
    """保存一次运行是否收到用户停止请求。

    注意：这是可变状态对象，不能用 ``==`` 判断"相等"（旧实现误加了
    ``@dataclass``，会让任意两个实例被判为相等）。
    """

    __slots__ = ("_event", "_requested")

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


_CURRENT: ContextVar[WorkflowCancellation | None] = ContextVar(
    "research_cancellation", default=None
)


def current_cancellation() -> WorkflowCancellation | None:
    """返回当前异步上下文绑定的取消信号；未绑定时为 ``None``。"""
    return _CURRENT.get()


def bind_cancellation(cancellation: WorkflowCancellation | None) -> Token:
    """绑定取消信号到当前上下文，返回用于恢复的 token。"""
    return _CURRENT.set(cancellation)


def reset_cancellation(token: Token) -> None:
    """恢复绑定前的上下文。"""
    _CURRENT.reset(token)


@contextmanager
def cancellation_scope(
    cancellation: WorkflowCancellation | None,
) -> Iterator[WorkflowCancellation | None]:
    """在本异步上下文内绑定取消信号，退出时恢复原值。"""
    token = _CURRENT.set(cancellation)
    try:
        yield cancellation
    finally:
        _CURRENT.reset(token)


__all__ = [
    "WorkflowCancellation",
    "bind_cancellation",
    "cancellation_scope",
    "current_cancellation",
    "reset_cancellation",
]
