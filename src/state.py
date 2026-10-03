"""State 定义模块：声明多智能体工作流的共享状态结构。

设计原则（参考 shopkeeper-agent 的分层 State）：
- 主 State 只放控制字段和请求入口
- 业务子类型各自独立，便于单独测试和维护
- 避免一个超大 TypedDict 难以理解和扩展的问题
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# 业务子状态类型
# ---------------------------------------------------------------------------


class SearchQueryPlan(TypedDict, total=False):
    """单次搜索计划中一个查询项的完整描述。"""

    section_id: str
    query: str
    source_preference: str  # "web" | "local" | "hybrid"
    reason: str


class OutlineSection(TypedDict, total=False):
    """大纲中单个章节的结构化描述。"""

    id: str
    title: str
    description: str
    section_type: str
    requires_data: bool
    requires_chart: bool
    priority: int
    search_queries: list[str]
    status: str


class EvidenceItem(TypedDict, total=False):
    """一条经过 LLM 结构化后的证据记录。"""

    source_id: str
    source_type: str  # "web" | "local"
    title: str
    url: str
    doc_id: str
    snippet: str
    domain: str
    supports_questions: list[str]
    reliability_score: float
    reliability_reason: str
    notes: str


class AuditFlag(TypedDict, total=False):
    """审计标记：低置信度 / 冲突 / 缺少证据。"""

    type: str  # "low_confidence" | "conflict" | "missing_evidence"
    target: str
    reason: str


class Finding(TypedDict, total=False):
    """一条结论性发现。"""

    claim_id: str
    claim: str
    confidence: str  # "high" | "medium" | "low"
    source_ids: list[str]


class SourceIndexItem(TypedDict, total=False):
    """参考资料索引中的一条条目。"""

    source_id: str
    label: str
    locator: str
    source_type: str


class RetrievalStats(TypedDict, total=False):
    """一次检索阶段的统计数据。"""

    query_count: int
    raw_count: int
    kept_count: int
    dropped_count: int


class QueryTrace(TypedDict, total=False):
    """单次检索查询的执行追踪记录。"""

    iteration: int
    plan_step: int
    query: str
    section_id: str
    reason: str
    source_preference: str
    raw_count: int
    raw_records: list[dict]
    kept_source_ids: list[str]
    rejected_source_ids: list[str]
    rejected_count: int


# ---------------------------------------------------------------------------
# 主 State
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """LangGraph 共享状态，覆盖一次深度调研的完整生命周期。

    字段按职责分组注释，便于新增节点时快速定位需要读写的字段。
    """

    # -- 请求入口 --
    query: str  # 用户原始问题
    user_id: str
    tenant_id: str
    memory_context: str  # 从短期/长期记忆中注入的上下文

    # -- 控制流 --
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str  # "direct" | "multiagent"
    phase: str  # 当前所处阶段标识
    iteration: int  # 反思循环次数
    max_iterations: int

    # -- 查询改写阶段产物 --
    rewritten_queries: list[str]  # 经 LLM 改写后的搜索词列表

    # -- 规划阶段产物 --
    objective: str
    outline: list[OutlineSection]  # 结构化大纲
    sub_questions: list[str]  # 拆解后的子问题
    research_questions: list[str]  # 研究问题列表
    search_plan: list[SearchQueryPlan]  # 搜索计划
    budget: dict  # token/轮次预算

    # -- 检索阶段产物 --
    web_search: str  # Web Scout 返回的摘要文本
    local_rag: str  # Local Scout 返回的摘要文本
    web_evidence: list[EvidenceItem]  # 网页证据列表
    local_evidence: list[EvidenceItem]  # 本地证据列表
    web_retrieval_stats: RetrievalStats
    local_retrieval_stats: RetrievalStats
    web_search_trace: list[QueryTrace]
    local_rag_trace: list[QueryTrace]

    # -- 证据裁判阶段产物 --
    deep_dive: str  # Evidence Judge 摘要
    evidence_pool: list[EvidenceItem]  # 综合证据池
    audit_flags: list[AuditFlag]  # 审计标记
    source_index: list[SourceIndexItem]  # 来源索引
    rerank_stats: dict  # Rerank 重排序统计信息

    # -- 分析阶段产物 --
    analysis: str  # Analyst 摘要
    findings: list[Finding]  # 结论列表
    claim_map: list[dict]  # claim_id -> source_ids 映射
    needs_more_research: bool
    missing_gaps: list[str]

    # -- 补搜阶段产物 --
    supplementary_queries: list[SearchQueryPlan]

    # -- 写作阶段产物 --
    draft: str
    final: str
    code: str  # CodeAgent 输出（可选）
