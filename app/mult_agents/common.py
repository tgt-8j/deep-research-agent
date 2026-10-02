"""多智能体层通用工具：颜色输出、超时调用等。"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger("mult_agents")

# ANSI 颜色码（仅 CLI 模式使用）
ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def colorize(text: str, color: str) -> str:
    """为文本添加 ANSI 颜色码，NO_COLOR 环境变量时返回原文。"""
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{ANSI['reset']}"


F = TypeVar("F", bound=Callable[..., Any])


def with_timeout(func: F, timeout_seconds: float = 30.0) -> F:
    """装饰器：为同步函数添加异步超时保护。

    用法：
        @with_timeout(timeout_seconds=30)
        def my_sync_function(...): ...

        # 在 async 上下文中调用：
        result = await asyncio.to_thread(my_sync_function, ...)
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("%s 调用超时（%.0fs）", func.__name__, timeout_seconds)
            raise
    return wrapper  # type: ignore[return-value]
