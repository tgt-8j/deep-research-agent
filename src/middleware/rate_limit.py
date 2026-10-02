"""API 限流中间件：基于 token bucket 算法限制请求速率。

防止恶意用户耗尽 LLM token 预算，保护后端服务稳定性。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response, HTTPException

logger = logging.getLogger("research.middleware.rate_limit")


class TokenBucket:
    """Token Bucket 限流器。

    原理：每个用户有一个桶，每秒补充固定数量的 token。
    每次请求消耗 1 个 token，桶为空时拒绝请求。
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒补充的 token 数
            capacity: 桶的最大容量（突发上限）
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """尝试获取 token，成功返回 True，失败返回 False。"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def remaining(self) -> float:
        """返回剩余 token 数。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self._capacity, self._tokens + elapsed * self._rate)


class RateLimiter:
    """基于用户 ID 的限流管理器。"""

    def __init__(self, default_rate: float = 2.0, default_capacity: int = 10):
        self._default_rate = default_rate
        self._default_capacity = default_capacity
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, user_id: str) -> TokenBucket:
        """获取或创建用户的 token bucket。"""
        if user_id not in self._buckets:
            self._buckets[user_id] = TokenBucket(
                rate=self._default_rate,
                capacity=self._default_capacity,
            )
        return self._buckets[user_id]

    async def check(self, user_id: str, tokens: int = 1) -> bool:
        """检查用户是否超过限流。"""
        bucket = self._get_bucket(user_id)
        return await bucket.acquire(tokens)


# 全局限流器实例
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """获取全局限流器单例。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """限流中间件。

    对 POST /api/v1/research/* 接口进行限流：
    - 默认限制：每秒 2 个请求，突发上限 10 个
    - 通过 user_id 区分不同用户
    """
    # 只限流研究接口
    if not request.url.path.startswith("/api/v1/research/"):
        return await call_next(request)

    # 从请求体或 header 获取 user_id
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        # 匿名请求更严格限制
        user_id = f"anon_{request.client.host}" if request.client else "anon"

    limiter = get_rate_limiter()
    allowed = await limiter.check(user_id)

    if not allowed:
        logger.warning("Rate limit exceeded | user=%s | path=%s", user_id, request.url.path)
        return Response(
            content='{"error": {"code": "RATE_LIMITED", "message": "请求过于频繁，请稍后重试"}}',
            status_code=429,
            media_type="application/json",
        )

    return await call_next(request)
