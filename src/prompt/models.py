"""Prompt 输出结构化验证：Pydantic 模型定义。

用于对 LLM 输出进行结构化验证，确保返回数据符合预期格式。
参考 shopkeeper-agent 的 models.py 模式。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidencePoolItem(BaseModel):
    """证据池中的一条记录。"""

    source_id: str = Field(..., description="来源ID，如 WEB1_1-1、LOC1_1-3")
    source_type: str = Field(..., description="来源类型: web | local")
    title: str = Field(..., description="标题")
    snippet: str = Field("", description="摘要片段")
    url: str = Field("", description="网址（本地知识库可为空）")
    doc_id: str = Field("", description="文档ID（网页搜索可为空）")
    reliability_score: float = Field(0.5, ge=0.0, le=1.0, description="可信度评分 0-1")
    reliability_reason: str = Field("", description="评分理由")
    supports_questions: list[str] = Field(
        default_factory=list, description="支撑的子问题列表"
    )
    notes: str = Field("", description="备注")


class AuditFlag(BaseModel):
    """审计标记：低置信度 / 冲突 / 缺少证据。"""

    type: str = Field(
        ..., description="类型: low_confidence | conflict | missing_evidence"
    )
    target: str = Field(..., description="目标问题或证据ID")
    reason: str = Field(..., description="原因说明")


class SourceIndexItem(BaseModel):
    """参考资料索引中的一条条目。"""

    source_id: str = Field(..., description="来源ID")
    label: str = Field(..., description="显示标签")
    locator: str = Field("", description="定位信息（URL或文档路径）")
    source_type: str = Field("source", description="来源类型")


class EvidencePool(BaseModel):
    """EvidenceJudge 输出结构。"""

    summary: str = Field(..., description="裁判摘要")
    evidence_pool: list[EvidencePoolItem] = Field(
        default_factory=list, description="证据池"
    )
    audit_flags: list[AuditFlag] = Field(default_factory=list, description="审计标记")
    source_index: list[SourceIndexItem] = Field(
        default_factory=list, description="来源索引"
    )


class FindingItem(BaseModel):
    """一条结论性发现。"""

    claim_id: str = Field(..., description="结论ID，如 c_1")
    claim: str = Field(..., description="结论内容")
    confidence: str = Field(..., description="置信度: high | medium | low")
    source_ids: list[str] = Field(default_factory=list, description="支撑来源ID列表")


class AnalysisOutput(BaseModel):
    """Analyst 输出结构。"""

    analysis_summary: str = Field(..., description="分析摘要")
    needs_more_research: bool = Field(False, description="是否需要更多研究")
    missing_gaps: list[str] = Field(default_factory=list, description="信息缺口列表")
    findings: list[FindingItem] = Field(default_factory=list, description="结论列表")
    claim_map: list[dict] = Field(
        default_factory=list, description="claim_id -> source_ids 映射"
    )
    next_actions: list[str] = Field(default_factory=list, description="下一步行动建议")


class ReflectionOutput(BaseModel):
    """ResearchPlanner 输出结构。"""

    reflection_summary: str = Field(..., description="反思摘要")
    supplementary_queries: list[dict] = Field(
        default_factory=list,
        description="补充搜索计划列表",
    )


class IntentOutput(BaseModel):
    """IntentRouter 输出结构。"""

    route: str = Field(..., description="路由结果: direct | multiagent")
    reason: str = Field(..., description="路由理由")


class QueryRewriteOutput(BaseModel):
    """QueryRewriter 输出结构。"""

    rewritten_queries: list[str] = Field(
        default_factory=list, description="改写后的查询列表"
    )
