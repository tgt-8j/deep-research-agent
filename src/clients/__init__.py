"""统一客户端管理器。

参考 shopkeeper-agent 的 clients/ 目录：
- 统一管理外部服务的连接池
- 延迟初始化（首次使用时创建）
- 统一的关闭逻辑
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("research.clients")


class ClientManager:
    """通用客户端管理器基类。

    所有外部服务客户端都应继承此类，统一管理生命周期。
    """

    def __init__(self, name: str):
        self.name = name
        self._client: Any | None = None

    @property
    def client(self) -> Any | None:
        return self._client

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def init(self, **kwargs) -> None:
        """初始化客户端。子类应重写此方法。"""
        raise NotImplementedError

    async def close(self) -> None:
        """关闭客户端。子类应重写此方法。"""
        self._client = None
        logger.info("[ClientManager] %s closed", self.name)


class MilvusClientManager(ClientManager):
    """Milvus 向量数据库客户端管理器。"""

    def __init__(self, host: str, port: int, collection: str, enable: bool = True):
        super().__init__("milvus")
        self._host = host
        self._port = port
        self._collection = collection
        self._enable = enable
        self._rag_system = None

    def init(self, api_key: str) -> None:
        if not self._enable:
            logger.info("[Milvus] disabled by config")
            return
        try:
            from ..retrieval.knowledge_base import create_knowledge_base_client

            self._rag_system = create_knowledge_base_client(
                milvus_host=self._host,
                milvus_port=self._port,
                collection_name=self._collection,
                enable_milvus=True,
                api_key=api_key,
            )
            self._client = self._rag_system
            logger.info(
                "[Milvus] connected | host=%s:%d collection=%s",
                self._host,
                self._port,
                self._collection,
            )
        except Exception as e:
            logger.warning("[Milvus] init failed: %s", e)

    async def close(self) -> None:
        self._rag_system = None
        self._client = None


class SessionStoreManager(ClientManager):
    """会话存储管理器。"""

    def __init__(self, storage_root: str = "./data/sessions"):
        super().__init__("session_store")
        self._storage_root = storage_root

    def init(self) -> None:
        from ..persistence.session_store import SessionStore

        self._client = SessionStore(storage_root=self._storage_root)
        logger.info("[SessionStore] initialized | root=%s", self._storage_root)


# =====================================================================
# 全局管理器单例
# =====================================================================

_milvus_manager: MilvusClientManager | None = None
_session_manager: SessionStoreManager | None = None


def get_milvus_manager(
    host: str, port: int, collection: str, enable: bool = True
) -> MilvusClientManager:
    """获取或创建 Milvus 客户端管理器单例。"""
    global _milvus_manager
    if _milvus_manager is None:
        _milvus_manager = MilvusClientManager(host, port, collection, enable)
    return _milvus_manager


def get_session_manager(storage_root: str = "./data/sessions") -> SessionStoreManager:
    """获取或创建会话存储管理器单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStoreManager(storage_root)
    return _session_manager


async def close_all_managers() -> None:
    """关闭所有客户端管理器。"""
    global _milvus_manager, _session_manager
    if _milvus_manager:
        await _milvus_manager.close()
        _milvus_manager = None
    if _session_manager:
        await _session_manager.close()
        _session_manager = None
