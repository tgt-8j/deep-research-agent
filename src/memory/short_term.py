"""短期记忆模块：基于 Redis 的对话记忆，TTL 自动过期。

设计原则：
- 支持 Redis 后端（生产）+ Memory 降级（开发/测试）
- TTL 控制内存占用，避免无限增长
- 提供统一的 add/get_recent/clear 接口
- 与 LangChain BaseMessage 兼容

使用方式：
    from src.memory.short_term import ShortTermMemory
    mem = ShortTermMemory(ttl_seconds=3600)
    mem.add(user_id="u1", message="你好")
    messages = mem.get_recent(user_id="u1", n=10)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger("research.memory.short_term")


class InMemoryShortTerm:
    """内存版短期记忆（降级方案）。"""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._storage: dict[str, list[dict[str, Any]]] = {}

    def _key(self, user_id: str) -> str:
        return f"st:{user_id}"

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self.ttl_seconds

    def add(
        self, user_id: str, message: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """添加消息到短期记忆。

        Args:
            user_id: 用户标识
            message: 消息内容
            metadata: 附加元数据

        Returns:
            消息 ID
        """
        import uuid

        msg_id = str(uuid.uuid4())[:8]
        entry = {
            "id": msg_id,
            "content": message,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat(),
        }
        key = self._key(user_id)
        if key not in self._storage:
            self._storage[key] = []
        self._storage[key].append(entry)
        logger.debug(
            "[InMemory] 添加消息到 %s，当前条数=%d", user_id, len(self._storage[key])
        )
        return msg_id

    def get_recent(self, user_id: str, n: int = 10) -> list[str]:
        """获取最近 N 条消息。

        Args:
            user_id: 用户标识
            n: 返回数量

        Returns:
            消息内容列表
        """
        key = self._key(user_id)
        if key not in self._storage:
            return []

        # 清理过期消息
        now = time.time()
        self._storage[key] = [
            e for e in self._storage[key] if not self._is_expired(e["timestamp"])
        ]

        # 返回最近 N 条
        recent = self._storage[key][-n:] if n > 0 else self._storage[key]
        return [e["content"] for e in recent]

    def clear(self, user_id: str) -> int:
        """清空指定用户的短期记忆。

        Args:
            user_id: 用户标识

        Returns:
            删除的消息数量
        """
        key = self._key(user_id)
        if key in self._storage:
            count = len(self._storage[key])
            del self._storage[key]
            logger.info("[InMemory] 清空 %s 的记忆，删除 %d 条", user_id, count)
            return count
        return 0

    def list_users(self) -> list[str]:
        """列出所有有记忆的用户。"""
        # 清理前缀 "st:"
        return [k.replace(self._key(""), "") for k in self._storage.keys()]


try:
    import redis

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False
    logger.warning("[short_term] redis 未安装，使用内存降级方案")


class RedisShortTerm:
    """Redis 版短期记忆（生产方案）。"""

    KEY_PREFIX = "ma:st:"
    TIMESTAMP_KEY_PREFIX = "ma:st:ts:"

    def __init__(
        self, redis_url: str = "redis://127.0.0.1:6379/0", ttl_seconds: int = 3600
    ):
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._redis_url = redis_url
        self._connect()

    def _connect(self) -> None:
        """连接到 Redis。"""
        if not _HAS_REDIS:
            logger.warning("[RedisShortTerm] redis 模块不可用")
            return
        try:
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
            self._client.ping()
            logger.info("[RedisShortTerm] 连接成功: %s", self._redis_url)
        except Exception as e:
            logger.warning("[RedisShortTerm] 连接失败: %s，使用内存降级", e)
            self._client = None

    def _key(self, user_id: str) -> str:
        return f"{self.KEY_PREFIX}{user_id}"

    def _is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._client = None
            return False

    def add(
        self, user_id: str, message: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """添加消息到 Redis。"""
        if not self._is_connected():
            logger.warning("[RedisShortTerm] 连接不可用，降级到内存")
            return self._fallback.add(user_id, message, metadata)

        import uuid

        msg_id = str(uuid.uuid4())[:8]
        entry = json.dumps(
            {
                "id": msg_id,
                "content": message,
                "metadata": metadata or {},
                "timestamp": time.time(),
                "created_at": datetime.now().isoformat(),
            }
        )
        key = self._key(user_id)
        self._client.rpush(key, entry)
        self._client.expire(key, self.ttl_seconds)
        logger.debug(
            "[Redis] 添加消息到 %s，当前条数=%d", user_id, self._client.llen(key)
        )
        return msg_id

    def get_recent(self, user_id: str, n: int = 10) -> list[str]:
        """获取最近 N 条消息。"""
        if not self._is_connected():
            return self._fallback.get_recent(user_id, n)

        key = self._key(user_id)
        raw_messages = self._client.lrange(key, -n, -1) or []
        messages = []
        for raw in raw_messages:
            try:
                entry = json.loads(raw)
                # 检查是否过期
                if time.time() - entry.get("timestamp", 0) > self.ttl_seconds:
                    continue
                messages.append(entry["content"])
            except json.JSONDecodeError:
                continue
        return messages

    def clear(self, user_id: str) -> int:
        """清空指定用户的短期记忆。"""
        if not self._is_connected():
            return self._fallback.clear(user_id)

        key = self._key(user_id)
        count = self._client.llen(key) if self._client.exists(key) else 0
        self._client.delete(key)
        logger.info("[Redis] 清空 %s 的记忆，删除 %d 条", user_id, count)
        return count

    def list_users(self) -> list[str]:
        """列出所有有记忆的用户。"""
        if not self._is_connected():
            return self._fallback.list_users()

        pattern = f"{self.KEY_PREFIX}*"
        keys = self._client.keys(pattern) or []
        return [k.replace(self.KEY_PREFIX, "") for k in keys]

    @property
    def is_alive(self) -> bool:
        """检查 Redis 是否可用。"""
        return self._is_connected()


class ShortTermMemory:
    """短期记忆主类：支持 Redis + Memory 双后端自动切换。

    属性：
        backend: "redis" | "memory"
        ttl_seconds: TTL 过期时间（秒）
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        redis_url: str | None = None,
        backend: str = "auto",
    ):
        """初始化短期记忆。

        Args:
            ttl_seconds: TTL 过期时间，默认 1 小时
            redis_url: Redis URL（backend="redis" 或 "auto" 时启用）
            backend: 后端类型
                - "auto": 自动选择（有 Redis 用 Redis，否则用 Memory）
                - "redis": 强制使用 Redis
                - "memory": 强制使用内存
        """
        self.ttl_seconds = ttl_seconds
        self._fallback = InMemoryShortTerm(ttl_seconds=ttl_seconds)

        # 初始化后端
        if backend == "memory" or (backend == "auto" and not _HAS_REDIS):
            self._backend = "memory"
            self._redis = None
            logger.info("[ShortTermMemory] 使用内存后端")
        elif backend == "redis" or backend == "auto":
            if _HAS_REDIS and redis_url:
                try:
                    self._redis = RedisShortTerm(
                        redis_url=redis_url, ttl_seconds=ttl_seconds
                    )
                    if self._redis.is_alive:
                        self._backend = "redis"
                        logger.info("[ShortTermMemory] 使用 Redis 后端: %s", redis_url)
                    else:
                        self._backend = "memory"
                        self._redis = None
                        logger.warning("[ShortTermMemory] Redis 连接失败，降级到内存")
                except Exception as e:
                    self._backend = "memory"
                    self._redis = None
                    logger.warning(
                        "[ShortTermMemory] Redis 初始化失败: %s，使用内存", e
                    )
            else:
                self._backend = "memory"
                self._redis = None
                logger.info("[ShortTermMemory] 无 Redis URL，使用内存后端")
        else:
            raise ValueError(f"未知的 backend: {backend}")

    @property
    def backend(self) -> str:
        """当前使用的后端。"""
        return self._backend

    def add(
        self, user_id: str, message: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """添加消息到短期记忆。

        Args:
            user_id: 用户标识
            message: 消息内容
            metadata: 附加元数据

        Returns:
            消息 ID
        """
        if self._backend == "redis" and self._redis:
            return self._redis.add(user_id, message, metadata)
        return self._fallback.add(user_id, message, metadata)

    def get_recent(self, user_id: str, n: int = 10) -> list[str]:
        """获取最近 N 条消息。

        Args:
            user_id: 用户标识
            n: 返回数量

        Returns:
            消息内容列表
        """
        if self._backend == "redis" and self._redis:
            return self._redis.get_recent(user_id, n)
        return self._fallback.get_recent(user_id, n)

    def clear(self, user_id: str) -> int:
        """清空指定用户的短期记忆。

        Args:
            user_id: 用户标识

        Returns:
            删除的消息数量
        """
        if self._backend == "redis" and self._redis:
            return self._redis.clear(user_id)
        return self._fallback.clear(user_id)

    def list_users(self) -> list[str]:
        """列出所有有记忆的用户。"""
        if self._backend == "redis" and self._redis:
            return self._redis.list_users()
        return self._fallback.list_users()

    def is_redis_available(self) -> bool:
        """检查 Redis 是否可用。"""
        return (
            self._backend == "redis"
            and self._redis is not None
            and self._redis.is_alive
        )


# 便捷工厂函数
def create_short_term_memory(
    ttl_seconds: int = 3600,
    redis_url: str | None = None,
    backend: str = "auto",
) -> ShortTermMemory:
    """创建短期记忆实例的便捷函数。

    Args:
        ttl_seconds: TTL 过期时间
        redis_url: Redis URL
        backend: 后端类型

    Returns:
        ShortTermMemory 实例
    """
    return ShortTermMemory(
        ttl_seconds=ttl_seconds,
        redis_url=redis_url,
        backend=backend,
    )


__all__ = [
    "InMemoryShortTerm",
    "RedisShortTerm",
    "ShortTermMemory",
    "create_short_term_memory",
]
