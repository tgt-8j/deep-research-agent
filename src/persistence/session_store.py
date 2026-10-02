"""会话持久化：SQLite + 文件系统混合存储。

参考 Paper-Agent 的 SessionStoreBackend，实现：
- SQLite 存储会话元数据和事件序列
- 文件系统存储中间产物（检索结果、大纲、报告草稿等）
- 支持从 Checkpoint 恢复中断的执行
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("research.persistence")


class Checkpoint:
    """一次工作流运行的检查点快照。

    保存当前 Stage 的完整状态，用于：
    1. 中断恢复：用户重新发起请求时从断点继续
    2. 调试：查看某次运行的中间状态
    """

    def __init__(
        self,
        session_key: str,
        turn_id: str,
        stage: str,
        state_snapshot: Dict[str, Any],
        error: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        self.session_key = session_key
        self.turn_id = turn_id
        self.stage = stage          # 中断时的阶段名
        self.state_snapshot = state_snapshot
        self.error = error
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_key": self.session_key,
            "turn_id": self.turn_id,
            "stage": self.stage,
            "state_snapshot": self.state_snapshot,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            session_key=data["session_key"],
            turn_id=data["turn_id"],
            stage=data["stage"],
            state_snapshot=data["state_snapshot"],
            error=data.get("error"),
        )


class SessionStore:
    """基于 SQLite 的会话持久化存储。

    职责：
    - 创建/查询/删除会话
    - 追加/查询事件序列（SSE 事件的持久化备份）
    - 保存/恢复 Checkpoint
    """

    def __init__(self, storage_root: str | Path = "./data/sessions"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_root / "session_store.db"
        self.checkpoints_dir = self.storage_root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """初始化数据库表结构。"""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session (
                    session_key TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    query TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    final_result TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_key) REFERENCES session(session_key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_log_session
                ON event_log (session_key, created_at)
            """)
            conn.commit()
        logger.info("SessionStore 初始化完成 | db=%s", self.db_path)

    # ------------------------------------------------------------------
    # 会话 CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_key: str,
        query: str,
        user_id: str,
        tenant_id: str,
        title: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO session
                    (session_key, title, query, user_id, tenant_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (session_key, title or query[:50], query, user_id, tenant_id, now, now),
            )
            conn.commit()

    def update_session_status(self, session_key: str, status: str, final_result: str = "") -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE session SET status = ?, final_result = ?, updated_at = ?
                WHERE session_key = ?
                """,
                (status, final_result, datetime.now().isoformat(), session_key),
            )
            conn.commit()

    def get_session(self, session_key: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM session WHERE session_key = ?", (session_key,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    def list_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT session_key, title, query, status, created_at FROM session "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 事件日志
    # ------------------------------------------------------------------

    def append_event(self, session_key: str, turn_id: str, event_type: str,
                     content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO event_log
                    (session_key, turn_id, event_type, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_key, turn_id, event_type, content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_events(self, session_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT event_type, content, metadata, created_at FROM event_log "
                "WHERE session_key = ? ORDER BY created_at ASC LIMIT ?",
                (session_key, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except json.JSONDecodeError:
                    pass
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint: Checkpoint) -> str:
        """保存 Checkpoint 到文件系统，并记录到数据库。"""
        cp_path = self.checkpoints_dir / f"{checkpoint.id}.json"
        cp_path.write_text(
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 同时记录摘要到数据库
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO event_log
                    (session_key, turn_id, event_type, content, metadata, created_at)
                VALUES (?, ?, 'checkpoint_saved', ?, ?, ?)
                """,
                (
                    checkpoint.session_key, checkpoint.turn_id,
                    json.dumps({"stage": checkpoint.stage, "checkpoint_id": checkpoint.id},
                               ensure_ascii=False),
                    json.dumps({"checkpoint_id": checkpoint.id, "stage": checkpoint.stage},
                               ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        logger.info("Checkpoint 已保存 | id=%s stage=%s session=%s",
                     checkpoint.id, checkpoint.stage, checkpoint.session_key)
        return checkpoint.id

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """从文件系统加载 Checkpoint。"""
        cp_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not cp_path.exists():
            return None
        try:
            data = json.loads(cp_path.read_text(encoding="utf-8"))
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Checkpoint 加载失败 | id=%s error=%s", checkpoint_id, e)
            return None

    def get_latest_checkpoint(self, session_key: str) -> Optional[Checkpoint]:
        """获取某个会话的最新 Checkpoint。"""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT metadata, created_at FROM event_log
                WHERE session_key = ? AND event_type = 'checkpoint_saved'
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_key,),
            ).fetchone()
        if not row:
            return None
        try:
            meta = json.loads(row["metadata"])
            return self.load_checkpoint(meta.get("checkpoint_id", ""))
        except Exception:
            return None

    def list_checkpoints(self, session_key: str) -> List[Dict[str, Any]]:
        """列出某个会话的所有 Checkpoint 摘要。"""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT metadata, created_at FROM event_log
                WHERE session_key = ? AND event_type = 'checkpoint_saved'
                ORDER BY created_at DESC
                """,
                (session_key,),
            ).fetchall()
        results = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"])
                results.append({
                    "checkpoint_id": meta.get("checkpoint_id"),
                    "stage": meta.get("stage"),
                    "created_at": row["created_at"],
                })
            except Exception:
                continue
        return results
