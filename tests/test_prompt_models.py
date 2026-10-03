"""Pydantic 模型验证测试：测试 LLM 输出的结构化验证。"""

from __future__ import annotations

import pytest

from src.prompt.models import (
    AnalysisOutput,
    AuditFlag,
    EvidencePool,
    EvidencePoolItem,
    FindingItem,
    IntentOutput,
    QueryRewriteOutput,
    ReflectionOutput,
    SourceIndexItem,
)


class TestEvidencePoolItem:
    """EvidencePoolItem 模型测试。"""

    def test_valid_item(self):
        """有效证据项应通过验证。"""
        item = EvidencePoolItem(
            source_id="WEB1_1-1",
            source_type="web",
            title="LangGraph 官方文档",
            snippet="LangGraph 是一个构建 Agent 的框架",
            url="https://langchain.com",
            reliability_score=0.95,
            reliability_reason="官方文档",
            supports_questions=["LangGraph 是什么？"],
        )
        assert item.source_id == "WEB1_1-1"
        assert item.source_type == "web"
        assert item.reliability_score == 0.95

    def test_score_boundary_low(self):
        """可信度边界检查：最低值。"""
        item = EvidencePoolItem(
            source_id="WEB-1",
            source_type="web",
            title="Test",
            reliability_score=0.0,
        )
        assert item.reliability_score == 0.0

    def test_score_boundary_high(self):
        """可信度边界检查：最高值。"""
        item = EvidencePoolItem(
            source_id="WEB-1",
            source_type="web",
            title="Test",
            reliability_score=1.0,
        )
        assert item.reliability_score == 1.0

    def test_score_out_of_range_low(self):
        """可信度超出范围应报错。"""
        with pytest.raises(ValueError):
            EvidencePoolItem(
                source_id="WEB-1",
                source_type="web",
                title="Test",
                reliability_score=-0.1,
            )

    def test_score_out_of_range_high(self):
        """可信度超出范围应报错。"""
        with pytest.raises(ValueError):
            EvidencePoolItem(
                source_id="WEB-1",
                source_type="web",
                title="Test",
                reliability_score=1.1,
            )

    def test_minimal_item(self):
        """最小化证据项（必填字段）。"""
        item = EvidencePoolItem(
            source_id="WEB-1",
            source_type="web",
            title="Test Title",
        )
        assert item.snippet == ""
        assert item.url == ""
        assert item.reliability_score == 0.5  # 默认值

    def test_from_json(self):
        """从 JSON 创建实例。"""
        data = {
            "source_id": "LOC1_1-1",
            "source_type": "local",
            "title": "企业内部文档",
            "snippet": "这是摘要",
            "doc_id": "doc_123",
            "reliability_score": 0.88,
            "supports_questions": ["问题1", "问题2"],
        }
        item = EvidencePoolItem(**data)
        assert item.source_id == "LOC1_1-1"
        assert len(item.supports_questions) == 2


class TestAuditFlag:
    """AuditFlag 模型测试。"""

    def test_valid_flag(self):
        """有效审计标记。"""
        flag = AuditFlag(
            type="low_confidence",
            target="问题1",
            reason="证据可信度低于阈值",
        )
        assert flag.type == "low_confidence"

    def test_conflict_flag(self):
        """冲突审计标记。"""
        flag = AuditFlag(
            type="conflict",
            target="c_1",
            reason="证据来源之间存在矛盾",
        )
        assert flag.type == "conflict"

    def test_missing_evidence_flag(self):
        """缺少证据审计标记。"""
        flag = AuditFlag(
            type="missing_evidence",
            target="子问题2",
            reason="无直接证据支撑该问题",
        )
        assert flag.type == "missing_evidence"


class TestSourceIndexItem:
    """SourceIndexItem 模型测试。"""

    def test_valid_item(self):
        """有效来源索引条目。"""
        item = SourceIndexItem(
            source_id="WEB1_1-1",
            label="LangGraph 官方文档",
            locator="https://langchain.com",
            source_type="web",
        )
        assert item.source_type == "web"

    def test_local_item(self):
        """本地知识库来源。"""
        item = SourceIndexItem(
            source_id="LOC1_1-1",
            label="企业内部知识",
            locator="doc://internal/kb/123",
            source_type="local",
        )
        assert item.source_type == "local"


class TestEvidencePool:
    """EvidencePool 模型测试。"""

    def test_empty_pool(self):
        """空证据池。"""
        pool = EvidencePool(summary="暂无证据")
        assert pool.evidence_pool == []
        assert pool.audit_flags == []
        assert pool.source_index == []

    def test_full_pool(self):
        """完整证据池。"""
        pool = EvidencePool(
            summary="裁判完成",
            evidence_pool=[
                EvidencePoolItem(
                    source_id="WEB-1",
                    source_type="web",
                    title="Test",
                    reliability_score=0.8,
                ),
            ],
            audit_flags=[
                AuditFlag(type="low_confidence", target="t1", reason="score < 0.5"),
            ],
            source_index=[
                SourceIndexItem(
                    source_id="WEB-1", label="Test", locator="https://test.com"
                ),
            ],
        )
        assert len(pool.evidence_pool) == 1
        assert len(pool.audit_flags) == 1
        assert len(pool.source_index) == 1

    def test_from_json_dict(self):
        """从字典创建证据池。"""
        data = {
            "summary": "测试摘要",
            "evidence_pool": [
                {
                    "source_id": "WEB-1",
                    "source_type": "web",
                    "title": "Test",
                    "reliability_score": 0.7,
                }
            ],
            "audit_flags": [],
            "source_index": [],
        }
        pool = EvidencePool(**data)
        assert pool.summary == "测试摘要"
        assert pool.evidence_pool[0].reliability_score == 0.7

    def test_validate_json_string(self):
        """从 JSON 字符串验证。"""
        import json

        data = {
            "summary": "test",
            "evidence_pool": [{"source_id": "W-1", "source_type": "web", "title": "T"}],
        }
        json_str = json.dumps(data)
        parsed = EvidencePool.model_validate_json(json_str)
        assert parsed.evidence_pool[0].source_id == "W-1"


class TestFindingItem:
    """FindingItem 模型测试。"""

    def test_valid_finding(self):
        """有效结论项。"""
        finding = FindingItem(
            claim_id="c_1",
            claim="LangGraph 支持状态图",
            confidence="high",
            source_ids=["WEB1_1-1", "LOC1_1-1"],
        )
        assert finding.claim_id == "c_1"
        assert finding.confidence == "high"

    def test_all_confidence_levels(self):
        """所有置信度级别。"""
        for level in ["high", "medium", "low"]:
            finding = FindingItem(
                claim_id=f"c_{level}",
                claim="Test",
                confidence=level,
            )
            assert finding.confidence == level


class TestAnalysisOutput:
    """AnalysisOutput 模型测试。"""

    def test_valid_output(self):
        """有效分析输出。"""
        output = AnalysisOutput(
            analysis_summary="分析完成",
            needs_more_research=False,
            missing_gaps=[],
            findings=[
                FindingItem(claim_id="c_1", claim="Test", confidence="high"),
            ],
            claim_map=[{"claim_id": "c_1", "source_ids": ["W-1"]}],
        )
        assert not output.needs_more_research
        assert len(output.findings) == 1

    def test_needs_more_research(self):
        """需要更多研究的情况。"""
        output = AnalysisOutput(
            analysis_summary="证据不足",
            needs_more_research=True,
            missing_gaps=["关于性能的对比数据"],
            findings=[],
        )
        assert output.needs_more_research
        assert output.missing_gaps == ["关于性能的对比数据"]

    def test_empty_findings(self):
        """空结论列表。"""
        output = AnalysisOutput(
            analysis_summary="无结论",
            needs_more_research=False,
            findings=[],
        )
        assert output.findings == []


class TestReflectionOutput:
    """ReflectionOutput 模型测试。"""

    def test_valid_output(self):
        """有效反思输出。"""
        output = ReflectionOutput(
            reflection_summary="发现信息缺口",
            supplementary_queries=[
                {
                    "section_id": "gap_1",
                    "query": "LangGraph 性能优化",
                    "source_preference": "hybrid",
                    "reason": "补搜性能数据",
                },
            ],
        )
        assert len(output.supplementary_queries) == 1

    def test_empty_queries(self):
        """空补充查询。"""
        output = ReflectionOutput(
            reflection_summary="无需补搜",
            supplementary_queries=[],
        )
        assert output.supplementary_queries == []


class TestIntentOutput:
    """IntentOutput 模型测试。"""

    def test_direct_route(self):
        """直接回答路由。"""
        output = IntentOutput(route="direct", reason="简单问候")
        assert output.route == "direct"

    def test_multiagent_route(self):
        """多智能体路由。"""
        output = IntentOutput(route="multiagent", reason="需要深度调研")
        assert output.route == "multiagent"


class TestQueryRewriteOutput:
    """QueryRewriteOutput 模型测试。"""

    def test_valid_output(self):
        """有效改写输出。"""
        output = QueryRewriteOutput(
            rewritten_queries=[
                "LangGraph 是什么",
                "LangGraph 使用教程",
                "LangGraph 示例",
            ],
        )
        assert len(output.rewritten_queries) == 3

    def test_empty_queries(self):
        """空查询列表。"""
        output = QueryRewriteOutput(rewritten_queries=[])
        assert output.rewritten_queries == []


class TestPydanticValidationIntegration:
    """与节点的集成测试。"""

    def test_deep_dive_validation(self):
        """模拟 deep_dive_node 的验证流程。"""
        import json

        from src.prompt.models import EvidencePool

        llm_output = {
            "summary": "证据裁判完成",
            "evidence_pool": [
                {
                    "source_id": "WEB1_1-1",
                    "source_type": "web",
                    "title": "LangGraph 文档",
                    "snippet": "介绍",
                    "url": "https://langchain.com",
                    "reliability_score": 0.9,
                    "reliability_reason": "官方文档",
                    "supports_questions": ["LangGraph 是什么"],
                }
            ],
            "audit_flags": [],
            "source_index": [],
        }

        # 使用 model_validate_json 验证
        pool = EvidencePool.model_validate_json(json.dumps(llm_output))
        assert len(pool.evidence_pool) == 1
        assert pool.evidence_pool[0].reliability_score == 0.9

    def test_analyze_validation(self):
        """模拟 analyze_node 的验证流程。"""
        import json

        from src.prompt.models import AnalysisOutput

        llm_output = {
            "analysis_summary": "分析完成",
            "needs_more_research": False,
            "missing_gaps": [],
            "findings": [
                {
                    "claim_id": "c_1",
                    "claim": "结论",
                    "confidence": "high",
                    "source_ids": ["W-1"],
                }
            ],
            "claim_map": [{"claim_id": "c_1", "source_ids": ["W-1"]}],
            "next_actions": [],
        }

        parsed = AnalysisOutput.model_validate_json(json.dumps(llm_output))
        assert not parsed.needs_more_research
        assert parsed.findings[0].confidence == "high"
