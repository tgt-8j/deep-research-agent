"""长期记忆模块：基于 SQLite 的用户偏好记忆 + 语义召回。

设计原则：
- 使用 SQLite 持久化存储，无需外部数据库
- LLM 从历史对话中提取用户偏好（如来源偏好、研究领域等）
- Embedding 语义召回，将相关偏好注入到 memory_context
- 支持多种记忆类型：语义记忆、情景记忆、程序性记忆

使用方式：
    from src.memory.long_term import LongTermMemory
    mem = LongTermMemory(db_path=":memory:")
    mem.store(user_id="u1", preferences=[{"key": "source_pref", "value": ".gov.cn"}])
    context = mem.recall(user_id="u1", query="搜索来源")
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any

logger = logging.getLogger("research.memory.long_term")

try:
    from langchain_community.embeddings import DashScopeEmbeddings

    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False
    logger.warning("[long_term] langchain_community.embeddings 不可用，禁用语义搜索")

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    logger.warning("[long_term] numpy 不可用，使用简单相似度")


# ---------------------------------------------------------------------------
# 嵌入工具
# ---------------------------------------------------------------------------


def _simple_embedding(text: str) -> list[float]:
    """简单字符级 embedding（无外部依赖时使用）。"""
    # 使用字符 ASCII 值的归一化作为简化 embedding
    if not text:
        return [0.0] * 64
    # bytes 迭代直接得到 int，不需要 ord()
    chars = text.encode("utf-8")[:64]
    vec = [c / 255.0 for c in chars]
    while len(vec) < 64:
        vec.append(0.0)
    return vec[:64]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算余弦相似度。"""
    if _HAS_NUMPY:
        a, b = np.array(vec1), np.array(vec2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """基于 SQLite 的长期记忆。

    存储用户偏好、历史任务、行为模式等，支持语义召回。

    记忆类型：
    - preference: 用户偏好（如来源偏好、研究领域）
    - fact: 事实性知识
    - episode: 历史任务执行记录
    - procedure: 标准操作流程（SOP）
    """

    def __init__(
        self,
        db_path: str | None = None,
        use_embedding: bool = True,
    ):
        """初始化长期记忆。

        Args:
            db_path: SQLite 数据库路径，默认 ":memory:"（内存）
            use_embedding: 是否使用 Embedding 语义搜索
        """
        if db_path is None:
            db_path = ":memory:"

        self.db_path = db_path
        self.use_embedding = use_embedding and _HAS_EMBEDDINGS
        self._conn: sqlite3.Connection | None = None

        # 初始化 Embedding 模型（如需）
        self._embedding_model = None
        if self.use_embedding:
            try:
                import os

                api_key = os.environ.get("DASHSCOPE_API_KEY", "")
                if api_key:
                    self._embedding_model = DashScopeEmbeddings(
                        model="text-embedding-v1",
                        dashscope_api_key=api_key,
                    )
                    logger.info("[LongTermMemory] Embedding 模型加载成功")
                else:
                    self.use_embedding = False
                    logger.warning(
                        "[LongTermMemory] DASHSCOPE_API_KEY 未设置，禁用语义搜索"
                    )
            except Exception as e:
                self.use_embedding = False
                logger.warning("[LongTermMemory] Embedding 初始化失败: %s", e)

        self._connect()
        self._init_tables()

    def _connect(self) -> None:
        """连接 SQLite 数据库。"""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        logger.debug("[LongTermMemory] 连接到 %s", self.db_path)

    def _init_tables(self) -> None:
        """初始化数据库表。"""
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
        )
        self._conn.commit()
        logger.debug("[LongTermMemory] 表初始化完成")

    # -----------------------------------------------------------------------
    # 核心接口
    # -----------------------------------------------------------------------

    def store(
        self,
        user_id: str,
        memories: list[dict[str, Any]],
        memory_type: str = "preference",
    ) -> list[str]:
        """存储记忆条目。

        Args:
            user_id: 用户标识
            memories: 记忆列表，每项包含 content 和可选的 metadata
            memory_type: 记忆类型（preference/fact/episode/procedure）

        Returns:
            存储的记忆 ID 列表
        """
        if self._conn is None:
            raise RuntimeError("数据库未连接")

        ids = []
        now = time.time()
        for item in memories:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            mem_id = f"{memory_type}:{user_id}:{int(now * 1000)}:{len(ids)}"

            # 生成 embedding
            embedding_json = None
            if self.use_embedding and self._embedding_model:
                try:
                    emb = self._embedding_model.embed_query(content)
                    embedding_json = json.dumps(emb)
                except Exception as e:
                    logger.warning("[LongTermMemory] Embedding 失败: %s", e)

            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, user_id, memory_type, content, embedding, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id,
                    user_id,
                    memory_type,
                    content,
                    embedding_json,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            ids.append(mem_id)

        self._conn.commit()
        logger.info(
            "[LongTermMemory] 存储 %d 条 %s 记忆给用户 %s",
            len(ids),
            memory_type,
            user_id,
        )
        return ids

    def recall(
        self,
        user_id: str,
        query: str,
        memory_type: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> str:
        """语义召回相关记忆，格式化为文本上下文。

        Args:
            user_id: 用户标识
            query: 查询词
            memory_type: 记忆类型过滤（可选）
            limit: 返回数量
            min_similarity: 最低相似度阈值

        Returns:
            格式化的记忆上下文文本
        """
        memories = self.search(
            user_id=user_id,
            query=query,
            memory_type=memory_type,
            limit=limit,
            min_similarity=min_similarity,
        )

        if not memories:
            return ""

        lines = ["[用户偏好与历史]", ""]
        for i, mem in enumerate(memories, 1):
            content = mem["content"]
            score = mem.get("similarity", 0.0)
            mtype = mem.get("memory_type", "unknown")
            lines.append(f"{i}. [{mtype}] (相似度={score:.2f}) {content}")

        return "\n".join(lines)

    def extract_preferences(self, messages: list[str]) -> list[dict[str, Any]]:
        """从历史消息中提取用户偏好。

        简化版本：基于关键词规则提取偏好。
        生产环境可使用 LLM 进行结构化提取。

        Args:
            messages: 历史消息列表

        Returns:
            提取的偏好列表
        """
        preferences = []

        # 关键词规则提取
        patterns = {
            "source_preference": [
                (r"\.gov\.cn", ".gov.cn"),
                (r"\.edu\.cn", ".edu.cn"),
                (r"官方来源", "official"),
                (r"权威来源", "authoritative"),
            ],
            "research_domain": [
                (r"(?:AI|人工智能|大模型|LLM)", "AI/LLM"),
                (r"(?:RAG|检索增强)", "RAG"),
                (r"(?:Agent|智能体)", "Agent"),
                (r"(?:LangGraph|LangChain)", "LangChain生态"),
            ],
            "language_preference": [
                (r"中文回答|用中文", "zh"),
                (r"English answer|in English", "en"),
            ],
        }

        combined_text = " ".join(messages)
        for pref_type, rules in patterns.items():
            for pattern, value in rules:
                if re.search(pattern, combined_text):
                    preferences.append(
                        {
                            "key": pref_type,
                            "value": value,
                            "confidence": 0.8,
                        }
                    )

        logger.info(
            "[LongTermMemory] 从 %d 条消息中提取 %d 条偏好",
            len(messages),
            len(preferences),
        )
        return preferences

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """获取用户完整档案。"""
        if self._conn is None:
            return {}

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT profile FROM user_profiles WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["profile"])
        return {}

    def update_profile(
        self,
        user_id: str,
        profile: dict[str, Any],
    ) -> None:
        """更新用户档案。"""
        if self._conn is None:
            return

        now = time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, profile, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile, updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(profile, ensure_ascii=False), now),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # 搜索接口
    # -----------------------------------------------------------------------

    def search(
        self,
        user_id: str,
        query: str,
        memory_type: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """搜索记忆。

        Args:
            user_id: 用户标识
            query: 查询词
            memory_type: 记忆类型过滤
            limit: 返回数量
            min_similarity: 最低相似度

        Returns:
            匹配的记忆列表，每项包含 content 和 similarity
        """
        if self._conn is None:
            return []

        cursor = self._conn.cursor()

        # 构建查询
        base_query = """
            SELECT id, user_id, memory_type, content, embedding, metadata, created_at
            FROM memories
            WHERE user_id = ?
        """
        params: list = [user_id]

        if memory_type:
            base_query += " AND memory_type = ?"
            params.append(memory_type)

        base_query += " ORDER BY created_at DESC"

        cursor.execute(base_query, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # 计算相似度
        query_embedding = self._get_embedding(query)
        results = []

        for row in rows:
            stored_embedding = (
                json.loads(row["embedding"]) if row["embedding"] else None
            )
            if stored_embedding is None:
                # 无 embedding 时使用简单文本匹配
                similarity = self._keyword_similarity(query, row["content"])
            else:
                similarity = _cosine_similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                results.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "memory_type": row["memory_type"],
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else {},
                        "similarity": round(similarity, 4),
                        "created_at": row["created_at"],
                    }
                )

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def list_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """列出用户的所有记忆。"""
        if self._conn is None:
            return []

        cursor = self._conn.cursor()
        query = """
            SELECT id, user_id, memory_type, content, metadata, created_at
            FROM memories
            WHERE user_id = ?
        """
        params = [user_id]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "memory_type": row["memory_type"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, memory_id: str) -> bool:
        """删除指定记忆。"""
        if self._conn is None:
            return False
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self, user_id: str) -> int:
        """清空用户所有记忆。"""
        if self._conn is None:
            return 0
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cursor.rowcount

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _get_embedding(self, text: str) -> list[float]:
        """获取文本的 embedding。"""
        if self.use_embedding and self._embedding_model:
            try:
                return self._embedding_model.embed_query(text)
            except Exception:
                pass
        return _simple_embedding(text)

    def _keyword_similarity(self, query: str, content: str) -> float:
        """简单关键词相似度（无 embedding 时使用）。"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words or not content_words:
            return 0.0
        intersection = query_words & content_words
        return len(intersection) / max(len(query_words), len(content_words))

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------


def create_long_term_memory(
    db_path: str | None = None,
    use_embedding: bool = True,
) -> LongTermMemory:
    """创建长期记忆实例。

    Args:
        db_path: 数据库路径
        use_embedding: 是否启用语义搜索

    Returns:
        LongTermMemory 实例
    """
    return LongTermMemory(db_path=db_path, use_embedding=use_embedding)


__all__ = [
    "LongTermMemory",
    "create_long_term_memory",
]
