"""Repository 数据访问层。

参考 shopkeeper-agent 的 repositories/ 目录：
- 将检索逻辑从 nodes 中提取到独立的 Repository 类
- 每个 Repository 封装一个存储后端（Milvus/Bocha Web Search 等）
- Node 只调用 Repository，不直接操作存储
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger("research.repositories")


class BaseRepository:
    """所有 Repository 的基类。"""

    def __init__(self, name: str):
        self.name = name

    def close(self) -> None:
        """释放资源。"""
        pass


class WebSearchRepository(BaseRepository):
    """网页搜索 Repository（封装 Bocha API）。

    职责：
    - 执行网页搜索
    - 过滤低质量结果
    - 返回结构化记录
    """

    def __init__(self, api_key: str = ""):
        super().__init__("web_search")
        self._api_key = api_key

    def search(self, query: str, count: int = 4) -> List[dict]:
        """执行网页搜索。"""
        from ..retrieval.web_search import bocha_web_search
        try:
            records = bocha_web_search(query, count=count, api_key=self._api_key)
            logger.info("[WebSearchRepo] query=%s | results=%d", query[:50], len(records))
            return records
        except Exception as e:
            logger.error("[WebSearchRepo] search failed: %s", e)
            return []

    def extract_content(self, url: str) -> str:
        """抓取 URL 内容。"""
        from ..retrieval.web_search import extract_url_content
        return extract_url_content(url)


class KnowledgeBaseRepository(BaseRepository):
    """知识库检索 Repository（封装 Milvus）。

    职责：
    - 执行向量相似度检索
    - 过滤不相关结果
    - 返回结构化证据
    """

    def __init__(self, kb_client: Any):
        super().__init__("knowledge_base")
        self._client = kb_client

    def search(self, query: str, limit: int = 4) -> List[dict]:
        """执行知识库检索。"""
        if self._client is None:
            logger.warning("[KBRepo] client not initialized, skipping")
            return []
        try:
            records = self._client.search(query, limit=limit)
            logger.info("[KBRepo] query=%s | results=%d", query[:50], len(records))
            return records
        except Exception as e:
            logger.error("[KBRepo] search failed: %s", e)
            return []


class SessionRepository(BaseRepository):
    """会话持久化 Repository（封装 SQLite）。

    职责：
    - 保存/恢复会话状态
    - 追加/查询事件日志
    - 管理 Checkpoint
    """

    def __init__(self, store: Any):
        super().__init__("session")
        self._store = store

    def save_checkpoint(self, checkpoint: Any) -> str:
        """保存 Checkpoint 并返回 ID。"""
        return self._store.save_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """加载 Checkpoint。"""
        return self._store.load_checkpoint(checkpoint_id)

    def append_event(self, session_key: str, event_type: str,
                     content: str, metadata: Optional[dict] = None) -> None:
        """追加事件日志。"""
        self._store.append_event(session_key, "turn_1", event_type, content, metadata)
