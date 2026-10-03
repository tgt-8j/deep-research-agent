"""实体模块：定义与业务领域相关的数据对象。

参考 shopkeeper-agent 的 entities/ 目录，将证据、来源等概念抽象为独立类，
便于类型检查和单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceRecord:
    """一条经过结构化后的证据记录。"""

    source_id: str
    source_type: str  # "web" | "local"
    title: str
    url: str = ""
    doc_id: str = ""
    snippet: str = ""
    domain: str = ""
    supports_questions: list[str] = field(default_factory=list)
    reliability_score: float = 0.0
    reliability_reason: str = ""
    notes: str = ""

    @property
    def locator(self) -> str:
        """返回证据的定位标识（URL 或文档 ID）。"""
        return self.url or self.doc_id or self.source_id


@dataclass
class SourceIndexEntry:
    """参考资料索引条目，用于报告末尾的引用列表生成。"""

    source_id: str
    label: str
    locator: str
    source_type: str

    @property
    def display_text(self) -> str:
        locator = self.locator or (
            "未提供链接" if self.source_type == "web" else "本地知识库"
        )
        return f"- [{self.source_id}] [{self.source_type}]: {self.label} | {locator}"


@dataclass
class AuditFlag:
    """审计标记：标识证据中的异常点。"""

    type: str  # "low_confidence" | "conflict" | "missing_evidence"
    target: str
    reason: str


@dataclass
class Finding:
    """一条结论性发现。"""

    claim_id: str
    claim: str
    confidence: str  # "high" | "medium" | "low"
    source_ids: list[str] = field(default_factory=list)


@dataclass
class OutlineSection:
    """大纲中的一个章节。"""

    id: str
    title: str
    description: str
    section_type: str = "mixed"
    requires_data: bool = False
    requires_chart: bool = False
    priority: int = 1
    search_queries: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class RetrievalStats:
    """一次检索阶段的统计数据。"""

    query_count: int = 0
    raw_count: int = 0
    kept_count: int = 0
    dropped_count: int = 0
