"""单元测试：State 定义和辅助函数。"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.state import EvidenceItem, Finding, OutlineSection, ResearchState


class TestResearchState:
    """测试 ResearchState 类型定义。"""

    def test_create_minimal_state(self):
        """最小状态创建。"""
        state: ResearchState = {
            "query": "测试问题",
            "user_id": "user_1",
            "tenant_id": "tenant_1",
            "messages": [],
            "intent": "",
            "phase": "initialized",
            "iteration": 0,
            "max_iterations": 3,
        }
        assert state["query"] == "测试问题"
        assert state["intent"] == ""

    def test_create_full_state(self):
        """完整状态创建。"""
        state: ResearchState = {
            "query": "LLM 框架对比",
            "user_id": "user_1",
            "tenant_id": "tenant_1",
            "memory_context": "用户偏好：中文",
            "messages": [],
            "intent": "multiagent",
            "phase": "planning",
            "iteration": 0,
            "max_iterations": 3,
            "objective": "LLM 框架对比分析",
            "outline": [
                {"id": "sec_1", "title": "概述", "description": "介绍", "priority": 1}
            ],
            "sub_questions": ["什么是LLM框架?", "各框架优缺点?"],
            "research_questions": ["LLM框架现状"],
            "search_plan": [
                {
                    "section_id": "sec_1",
                    "query": "LLM框架",
                    "source_preference": "hybrid",
                }
            ],
            "budget": {"max_rounds": 2, "max_sources": 10},
            "web_search": "网页检索结果",
            "local_rag": "本地检索结果",
            "web_evidence": [
                {"source_id": "WEB-1", "source_type": "web", "title": "test"}
            ],
            "local_evidence": [
                {"source_id": "LOC-1", "source_type": "local", "title": "test"}
            ],
            "evidence_pool": [],
            "audit_flags": [],
            "source_index": [],
            "analysis": "分析结论",
            "findings": [{"claim_id": "c_1", "claim": "test", "confidence": "high"}],
            "claim_map": [],
            "needs_more_research": False,
            "missing_gaps": [],
            "supplementary_queries": [],
            "draft": "# 研究报告\n\n内容...",
            "final": "# 研究报告\n\n内容...",
            "code": "",
            "web_retrieval_stats": {"query_count": 2, "raw_count": 10, "kept_count": 5},
            "local_retrieval_stats": {},
            "web_search_trace": [],
            "local_rag_trace": [],
        }
        assert state["intent"] == "multiagent"
        assert len(state["web_evidence"]) == 1
        assert state["findings"][0]["claim_id"] == "c_1"

    def test_state_type_check(self):
        """验证 State 是 TypedDict。"""
        from typing import get_type_hints

        hints = get_type_hints(ResearchState)
        assert "query" in hints
        assert "messages" in hints
        assert "final" in hints


class TestSubTypes:
    """测试子类型定义。"""

    def test_outline_section(self):
        section: OutlineSection = {
            "id": "sec_1",
            "title": "引言",
            "description": "背景介绍",
            "priority": 1,
        }
        assert section["id"] == "sec_1"

    def test_evidence_item(self):
        evidence: EvidenceItem = {
            "source_id": "WEB-1",
            "source_type": "web",
            "title": "测试标题",
            "url": "https://example.com",
            "snippet": "摘要内容",
            "reliability_score": 0.85,
        }
        assert evidence["reliability_score"] == 0.85

    def test_finding(self):
        finding: Finding = {
            "claim_id": "c_1",
            "claim": "LangGraph 是目前最流行的 Agent 框架",
            "confidence": "high",
            "source_ids": ["WEB1_1-1", "WEB1_1-2"],
        }
        assert finding["confidence"] == "high"
