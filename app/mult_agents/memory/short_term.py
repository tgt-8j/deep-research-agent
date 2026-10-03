"""
短期记忆模块

基于 LangGraph Checkpoint 实现，管理当前对话线程的上下文
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from .base import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger("mult_agents.memory")

try:
    import redis
except Exception:
    redis = None


class ConversationBuffer:
    """
    对话缓冲区
    
    管理单轮对话的消息历史，支持窗口裁剪和摘要生成
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        summary_threshold: int = 10,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.messages: list[BaseMessage] = []
        self.summary: str | None = None
        self.token_count: int = 0
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息到缓冲区"""
        self.messages.append(message)
        self._update_token_count()
        
        # 如果超过阈值，触发裁剪或摘要
        if len(self.messages) > self.max_messages:
            self._compress_messages()
    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        """批量添加消息"""
        for msg in messages:
            self.add_message(msg)
    
    def get_messages(
        self,
        include_summary: bool = True,
        last_n: int | None = None
    ) -> list[BaseMessage]:
        """
        获取消息列表
        
        Args:
            include_summary: 是否包含历史摘要
            last_n: 只返回最近 N 条消息
            
        Returns:
            消息列表
        """
        result = []
        
        # 添加摘要作为系统消息
        if include_summary and self.summary:
            result.append(SystemMessage(content=f"历史对话摘要：{self.summary}"))
        
        # 添加实际消息
        messages_to_return = self.messages
        if last_n:
            messages_to_return = self.messages[-last_n:]
        
        result.extend(messages_to_return)
        return result
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.messages = []
        self.summary = None
        self.token_count = 0
    
    def _update_token_count(self) -> None:
        """更新 token 计数（简化估算）"""
        # 简单估算：每个字符约 0.5 个 token
        total_chars = sum(len(str(msg.content)) for msg in self.messages)
        self.token_count = total_chars // 2
    
    def _compress_messages(self) -> None:
        """
        压缩消息历史
        
        策略：保留最近的消息，将旧消息生成摘要
        """
        if len(self.messages) <= self.summary_threshold:
            return
        
        # 保留最近的消息
        messages_to_summarize = self.messages[:-self.summary_threshold]
        self.messages = self.messages[-self.summary_threshold:]
        
        # 生成摘要（简化版本，实际可以调用 LLM）
        summary_parts = []
        for msg in messages_to_summarize:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            content_preview = str(msg.content)[:100]
            summary_parts.append(f"{role}: {content_preview}...")
        
        new_summary = "\n".join(summary_parts)
        
        if self.summary:
            self.summary = f"{self.summary}\n\n[更早的对话]\n{new_summary}"
        else:
            self.summary = new_summary
        
        logger.debug(f"消息历史已压缩，当前消息数: {len(self.messages)}")


class ShortTermMemory(BaseMemory):
    """
    短期记忆实现
    
    基于内存存储，管理当前线程的对话上下文和临时状态
    与 LangGraph 的 State 和 Checkpoint 集成
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,  # 默认 1 小时过期
        max_threads: int = 100,
    ):
        super().__init__(MemoryType.SHORT_TERM)
        self.ttl_seconds = ttl_seconds
        self.max_threads = max_threads
        
        # 存储结构: {thread_id: {"buffer": ConversationBuffer, "metadata": {}, "last_access": datetime}}
        self._storage: dict[str, dict[str, Any]] = {}
        self._checkpointer: BaseCheckpointSaver | None = None
    
    def set_checkpointer(self, checkpointer: BaseCheckpointSaver) -> None:
        """设置 LangGraph Checkpoint 存储"""
        self._checkpointer = checkpointer
    
    def get_or_create_buffer(self, thread_id: str) -> ConversationBuffer:
        """获取或创建对话缓冲区"""
        self._cleanup_expired()
        
        if thread_id not in self._storage:
            self._storage[thread_id] = {
                "buffer": ConversationBuffer(),
                "metadata": {},
                "created_at": datetime.now(),
                "last_access": datetime.now(),
            }
            logger.debug(f"为新线程 {thread_id} 创建短期记忆缓冲区")
        else:
            self._storage[thread_id]["last_access"] = datetime.now()
        
        return self._storage[thread_id]["buffer"]
    
    def add_message(
        self,
        thread_id: str,
        message: BaseMessage,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        添加消息到短期记忆
        
        Args:
            thread_id: 线程标识
            message: LangChain 消息对象
            metadata: 附加元数据
        """
        buffer = self.get_or_create_buffer(thread_id)
        buffer.add_message(message)
        
        if metadata:
            self._storage[thread_id]["metadata"].update(metadata)
        
        logger.debug(f"消息已添加到线程 {thread_id} 的短期记忆")
    
    def get_messages(
        self,
        thread_id: str,
        include_summary: bool = True,
        last_n: int | None = None
    ) -> list[BaseMessage]:
        """
        获取指定线程的消息历史
        
        Args:
            thread_id: 线程标识
            include_summary: 是否包含历史摘要
            last_n: 只返回最近 N 条
            
        Returns:
            消息列表
        """
        if thread_id not in self._storage:
            return []
        
        self._storage[thread_id]["last_access"] = datetime.now()
        buffer = self._storage[thread_id]["buffer"]
        return buffer.get_messages(include_summary=include_summary, last_n=last_n)
    
    def get_thread_metadata(self, thread_id: str) -> dict[str, Any]:
        """获取线程元数据"""
        if thread_id not in self._storage:
            return {}
        return self._storage[thread_id]["metadata"].copy()
    
    def update_thread_metadata(
        self,
        thread_id: str,
        metadata: dict[str, Any]
    ) -> None:
        """更新线程元数据"""
        self._storage[thread_id]["metadata"].update(metadata)
    
    def clear_thread(self, thread_id: str) -> bool:
        """清空指定线程的记忆"""
        if thread_id in self._storage:
            del self._storage[thread_id]
            logger.debug(f"线程 {thread_id} 的短期记忆已清空")
            return True
        return False
    
    def list_active_threads(self) -> list[str]:
        """列出所有活跃线程"""
        self._cleanup_expired()
        return list(self._storage.keys())
    
    # 实现 BaseMemory 抽象方法
    
    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目（短期记忆使用专用方法）"""
        thread_id = entry.thread_id or "default"
        buffer = self.get_or_create_buffer(thread_id)
        
        # 将内容转换为消息
        if isinstance(entry.content, str):
            message = HumanMessage(content=entry.content)
        elif isinstance(entry.content, dict):
            content = entry.content.get("content", "")
            role = entry.content.get("role", "human")
            if role == "ai":
                message = AIMessage(content=content)
            else:
                message = HumanMessage(content=content)
        else:
            message = HumanMessage(content=str(entry.content))
        
        buffer.add_message(message)
        
        # 更新元数据
        if entry.metadata:
            self._storage[thread_id]["metadata"].update(entry.metadata)
        
        return entry.id
    
    def get(self, memory_id: str) -> MemoryEntry | None:
        """获取指定 ID 的记忆（短期记忆不支持按 ID 获取）"""
        # 短期记忆不支持按 ID 获取，返回 None
        return None
    
    def search(
        self,
        query: str,
        user_id: str | None = None,
        namespace: str | None = None,
        limit: int = 5,
        **kwargs
    ) -> list[MemoryEntry]:
        """
        搜索短期记忆
        
        简单实现：返回最近的消息作为记忆条目
        """
        thread_id = namespace or "default"
        messages = self.get_messages(thread_id, include_summary=False)
        
        entries = []
        for msg in messages[-limit:]:
            entry = MemoryEntry(
                content=msg.content,
                memory_type=MemoryType.SHORT_TERM,
                thread_id=thread_id,
                user_id=user_id,
                metadata={"role": "ai" if isinstance(msg, AIMessage) else "human"},
            )
            entries.append(entry)
        
        return entries
    
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆（短期记忆不支持按 ID 删除）"""
        return False
    
    def clear(
        self,
        user_id: str | None = None,
        namespace: str | None = None
    ) -> int:
        """清除短期记忆"""
        if namespace:
            # 清除指定线程
            if self.clear_thread(namespace):
                return 1
            return 0
        
        # 清除所有
        count = len(self._storage)
        self._storage.clear()
        logger.info(f"已清除所有短期记忆，共 {count} 个线程")
        return count
    
    def list_namespaces(self, user_id: str | None = None) -> list[str]:
        """列出所有线程 ID 作为命名空间"""
        return self.list_active_threads()
    
    def _cleanup_expired(self) -> None:
        """清理过期的线程记忆"""
        now = datetime.now()
        expired_threads = []
        
        for thread_id, data in self._storage.items():
            last_access = data.get("last_access", data.get("created_at", now))
            if now - last_access > timedelta(seconds=self.ttl_seconds):
                expired_threads.append(thread_id)
        
        # 如果超过最大线程数，清理最旧的
        if len(self._storage) > self.max_threads:
            sorted_threads = sorted(
                self._storage.items(),
                key=lambda x: x[1].get("last_access", x[1].get("created_at", now))
            )
            threads_to_remove = len(self._storage) - self.max_threads
            expired_threads.extend([t[0] for t in sorted_threads[:threads_to_remove]])
        
        for thread_id in set(expired_threads):
            del self._storage[thread_id]
        
        if expired_threads:
            logger.debug(f"已清理 {len(set(expired_threads))} 个过期线程")


class RedisShortTermMemory(BaseMemory):
    """
    基于 Redis 的短期记忆实现

    使用 Redis List 存储对话消息，String 存储摘要。
    支持 TTL 过期、自动压缩、连接失败降级到内存。

    Key 模式:
        - 消息列表: ma:short:{tenant}:{user}:{thread}
        - 摘要:     ma:short:summary:{tenant}:{user}:{thread}
    """

    KEY_PREFIX = "ma:short"
    SUMMARY_PREFIX = "ma:short:summary"

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        ttl_seconds: int = 604800,
        max_messages: int = 30,
        summary_threshold: int = 20,
        failover_enabled: bool = True,
    ):
        super().__init__(MemoryType.SHORT_TERM)
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.failover_enabled = failover_enabled

        self._client = None
        self._fallback = ShortTermMemory(ttl_seconds=ttl_seconds)
        self._connection_ok = False
        self._last_error: str | None = None
        self._last_success: datetime | None = None
        self._summary_llm = None  # 由 MemoryManager 注入

        self._connect(redis_url)

    def _connect(self, redis_url: str) -> None:
        """连接 Redis，支持 root: 密码的 fallback"""
        if redis is None or not redis_url:
            return
        urls = [redis_url]
        if "redis://root:" in redis_url:
            urls.append(redis_url.replace("redis://root:", "redis://:"))
        for url in urls:
            try:
                client = redis.Redis.from_url(url, decode_responses=True)
                client.ping()
                self._client = client
                self._connection_ok = True
                self._last_success = datetime.now()
                logger.info("Redis 短期记忆连接成功: %s", url)
                return
            except Exception as exc:
                logger.debug("Redis 连接尝试失败: %s", exc)
        logger.warning("Redis 连接失败，将使用内存短期记忆降级")

    def _ensure_connected(self) -> bool:
        """检查连接状态，必要时尝试重连"""
        if self._client is None:
            return False
        try:
            self._client.ping()
            self._connection_ok = True
            self._last_success = datetime.now()
            return True
        except Exception:
            self._connection_ok = False
            self._last_error = f"Redis ping 失败: {self.redis_url}"
            if self.failover_enabled:
                logger.warning("Redis 连接中断，降级到内存短期记忆: %s", self._last_error)
                return False
            raise
        return False

    # ── 内部工具方法 ──────────────────────────────────────────────

    def _thread_key(self, tenant: str, user: str, thread: str) -> str:
        return f"{self.KEY_PREFIX}:{tenant}:{user}:{thread}"

    def _summary_key(self, tenant: str, user: str, thread: str) -> str:
        return f"{self.SUMMARY_PREFIX}:{tenant}:{user}:{thread}"

    @staticmethod
    def _serialize(message: BaseMessage) -> dict[str, str]:
        role_map = {
            HumanMessage: "human",
            AIMessage: "ai",
            SystemMessage: "system",
        }
        role = next((r for cls, r in role_map.items() if isinstance(message, cls)), "human")
        return {"role": role, "content": str(message.content)}

    @staticmethod
    def _deserialize(payload: dict[str, str]) -> BaseMessage:
        role = payload.get("role", "human")
        content = payload.get("content", "")
        if role == "ai":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    def _compress(self, tenant: str, user: str, thread: str) -> None:
        """压缩线程消息：超过阈值时生成摘要并保留最近消息"""
        if not self._ensure_connected():
            return
        key = self._thread_key(tenant, user, thread)
        summary_key = self._summary_key(tenant, user, thread)
        raw = self._client.lrange(key, 0, -1) or []
        if len(raw) <= self.max_messages:
            return

        parsed = [json.loads(item) for item in raw]
        split_at = len(parsed) - self.summary_threshold
        to_summarize = parsed[:split_at]
        keep = parsed[split_at:]

        existing_summary = self._client.get(summary_key) or ""
        new_summary = self._summarize(existing_summary, to_summarize)

        pipe = self._client.pipeline()
        pipe.delete(key)
        if keep:
            pipe.rpush(key, *[json.dumps(item, ensure_ascii=False) for item in keep])
        pipe.set(summary_key, new_summary, ex=self.ttl_seconds)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()
        logger.debug("Redis 短期记忆已压缩，当前消息数: %d", len(keep))

    def _summarize(self, existing: str, history: list[dict[str, str]]) -> str:
        """生成摘要：优先使用 LLM，否则截断拼接"""
        lines = [f"{item.get('role', 'human')}: {item.get('content', '')}" for item in history]
        text = "\n".join(lines)
        if self._summary_llm is None:
            combined = f"{existing}\n{text}".strip()
            return combined[-4000:] if combined else ""
        try:
            prompt = (
                "你是对话压缩引擎。请在保留事实、偏好、结论、待办和约束的前提下进行递归摘要。\n"
                f"已有摘要：{existing or '无'}\n"
                "新增历史：\n"
                f"{text}\n"
                "输出要求：100-300字，中文，结构紧凑。"
            )
            resp = self._summary_llm.invoke([HumanMessage(content=prompt)])
            return str(resp.content).strip()
        except Exception as exc:
            logger.warning("LLM 摘要失败，使用截断: %s", exc)
            combined = f"{existing}\n{text}".strip()
            return combined[-4000:] if combined else ""

    # ── 公开 API（BaseMemory 接口） ────────────────────────────────

    def is_alive(self) -> bool:
        """检查 Redis 连接是否可用"""
        if self._client is None:
            return False
        try:
            self._client.ping()
            self._connection_ok = True
            return True
        except Exception:
            self._connection_ok = False
            return False

    def save(self, entry: MemoryEntry) -> str:
        thread_id = entry.thread_id or "default"
        tenant = entry.metadata.get("tenant_id", "default_tenant")
        user = entry.metadata.get("user_id", "default_user")
        if self._ensure_connected():
            try:
                payload = self._serialize(
                    HumanMessage(content=str(entry.content))
                    if isinstance(entry.content, str)
                    else self._deserialize(entry.content)
                )
                key = self._thread_key(tenant, user, thread_id)
                self._client.rpush(key, json.dumps(payload, ensure_ascii=False))
                self._client.expire(key, self.ttl_seconds)
                self._compress(tenant, user, thread_id)
                self._last_success = datetime.now()
                return entry.id
            except Exception as exc:
                if self.failover_enabled:
                    logger.warning("Redis 写入失败，降级内存: %s", exc)
                    return self._fallback.save(entry)
                raise
        return self._fallback.save(entry)

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Redis 短期记忆不支持按 ID 查询，返回 None"""
        return None

    def search(
        self,
        query: str,
        user_id: str | None = None,
        namespace: str | None = None,
        limit: int = 5,
        **kwargs,
    ) -> list[MemoryEntry]:
        thread_id = namespace or "default"
        messages = self.get_messages(thread_id, include_summary=False)
        entries = []
        for msg in messages[-limit:]:
            entries.append(MemoryEntry(
                content=msg.content,
                memory_type=MemoryType.SHORT_TERM,
                thread_id=thread_id,
                user_id=user_id,
                metadata={"role": "ai" if isinstance(msg, AIMessage) else "human"},
            ))
        return entries

    def delete(self, memory_id: str) -> bool:
        return False

    def clear(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        if not self._ensure_connected():
            return self._fallback.clear(user_id=user_id, namespace=namespace)
        if namespace:
            # 清除指定线程
            keys = self._client.keys(f"{self.KEY_PREFIX}:*:*:{namespace}") or []
            summary_keys = self._client.keys(f"{self.SUMMARY_PREFIX}:*:*:{namespace}") or []
            if keys:
                self._client.delete(*keys)
            if summary_keys:
                self._client.delete(*summary_keys)
            return 1
        # 清除所有短期记忆
        if user_id:
            keys = self._client.keys(f"{self.KEY_PREFIX}:*:{user_id}:*") or []
            summary_keys = self._client.keys(f"{self.SUMMARY_PREFIX}:*:{user_id}:*") or []
        else:
            keys = self._client.keys(f"{self.KEY_PREFIX}:*:*:*") or []
            summary_keys = self._client.keys(f"{self.SUMMARY_PREFIX}:*:*:*") or []
        deleted = 0
        if keys:
            deleted += self._client.delete(*keys)
        if summary_keys:
            deleted += self._client.delete(*summary_keys)
        return deleted

    def list_namespaces(self, user_id: str | None = None) -> list[str]:
        if not self._ensure_connected():
            return self._fallback.list_active_threads()
        pattern = f"{self.KEY_PREFIX}:*:*:*"
        keys = self._client.keys(pattern) or []
        threads = {k.rsplit(":", 1)[-1] for k in keys if self.SUMMARY_PREFIX not in k}
        if user_id:
            threads = {t for t in threads if t}  # 进一步过滤（简化处理）
        return sorted(threads)

    # ── 便捷方法（被 Manager 调用） ────────────────────────────────

    def add_message(
        self,
        thread_id: str,
        message: BaseMessage,
        metadata: dict[str, Any] | None = None,
        user_id: str = "default_user",
        tenant_id: str = "default_tenant",
    ) -> None:
        metadata = metadata or {}
        metadata.update({"tenant_id": tenant_id, "user_id": user_id})
        entry = MemoryEntry(
            content=message.content,
            memory_type=MemoryType.SHORT_TERM,
            thread_id=thread_id,
            user_id=user_id,
            metadata=metadata,
        )
        self.save(entry)

    def get_messages(
        self,
        thread_id: str,
        include_summary: bool = True,
        last_n: int | None = None,
        user_id: str = "default_user",
        tenant_id: str = "default_tenant",
    ) -> list[BaseMessage]:
        if not self._ensure_connected():
            return self._fallback.get_messages(thread_id, include_summary, last_n)
        key = self._thread_key(tenant_id, user_id, thread_id)
        raw = self._client.lrange(key, 0, -1) or []
        if last_n:
            raw = raw[-last_n:]
        messages = [self._deserialize(json.loads(item)) for item in raw]
        if include_summary:
            summary_key = self._summary_key(tenant_id, user_id, thread_id)
            summary = self._client.get(summary_key) or ""
            if summary:
                messages = [SystemMessage(content=f"历史对话摘要：{summary}"), *messages]
        return messages

    def get_summary(
        self,
        thread_id: str,
        user_id: str = "default_user",
        tenant_id: str = "default_tenant",
    ) -> str:
        if not self._ensure_connected():
            return ""
        key = self._summary_key(tenant_id, user_id, thread_id)
        return self._client.get(key) or ""

    def is_short_term_empty(
        self,
        tenant: str,
        user_id: str,
        thread_id: str,
    ) -> bool:
        if not self._ensure_connected():
            return True
        key = self._thread_key(tenant, user_id, thread_id)
        return int(self._client.llen(key) or 0) == 0
