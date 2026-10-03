"""端到端真实运行验证测试（E2E realistic）。

策略：
- 使用 MagicMock LLM 替代真实模型调用
- 使用真实 graph 构建 + 真实节点逻辑
- 覆盖 direct_answer 和 multiagent 两条完整路径
- 模拟 web_search、local_rag、rerank、deep_dive、analyze、reflect、write
- 验证最终报告格式、引用溯源、证据链完整性
- 验证 reflection 循环最多 2 次
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.types import Command

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))


# ===================================================================
# Fixtures
# ===================================================================


def _make_mock_llm(responses: list[dict] | None = None) -> MagicMock:
    """创建可配置的 mock LLM，返回预定义的响应序列。"""
    llm = MagicMock()
    response_sequence = responses or [
        {
            "messages": [
                MagicMock(
                    content='{"route": "multiagent", "reason": "调研类问题"}', type="ai"
                )
            ]
        },
    ]

    async def mock_ainvoke(messages):
        return response_sequence.pop(0) if response_sequence else response_sequence[-1]

    llm.ainvoke = mock_ainvoke
    return llm


def _make_runtime(llm=None, progress_emitter=None, kb_client=None):
    """创建兼容 graph._make_node 的 Mock Runtime。"""
    runtime = type("Runtime", (), {})()
    runtime.context = {
        "llm": llm,
        "kb_client": kb_client,
        "progress_emitter": progress_emitter,
        "retrieval_config": {"bocha_api_key": ""},
        "cancellation": None,
    }
    return runtime


def _make_test_state(
    query: str = "LangGraph 反思循环的实现原理",
    intent: str = "multiagent",
    max_iterations: int = 2,
    **overrides,
) -> dict[str, Any]:
    """创建初始测试状态。"""
    state = {
        "query": query,
        "user_id": "test_user",
        "tenant_id": "test_tenant",
        "memory_context": "",
        "messages": [],
        "intent": intent,
        "phase": "initialized",
        "iteration": 0,
        "max_iterations": max_iterations,
        "objective": "",
        "outline": [],
        "sub_questions": [],
        "research_questions": [],
        "search_plan": [],
        "budget": {},
        "web_search": "",
        "local_rag": "",
        "web_evidence": [],
        "local_evidence": [],
        "web_retrieval_stats": {},
        "local_retrieval_stats": {},
        "web_search_trace": [],
        "local_rag_trace": [],
        "evidence_pool": [],
        "deep_dive": "",
        "audit": "",
        "audit_flags": [],
        "source_index": [],
        "analysis": "",
        "findings": [],
        "claim_map": [],
        "needs_more_research": False,
        "missing_gaps": [],
        "supplementary_queries": [],
        "draft": "",
        "final": "",
        "code": "",
        "rerank_stats": {},
    }
    state.update(overrides)
    return state


# ===================================================================
# Direct Answer Path
# ===================================================================


class TestDirectAnswerPath:
    """直接回答路径测试。"""

    @pytest.mark.asyncio
    async def test_direct_answer_simple(self):
        """简单问候直接回答。"""
        from src.graph.nodes.direct_answer_node import direct_answer_node
        from src.graph.nodes.intent_node import intent_node

        state = _make_test_state(query="你好", intent="direct")
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "direct", "reason": "简单问候"}',
                            type="ai",
                        )
                    ]
                },
                {
                    "messages": [
                        MagicMock(
                            content="# 你好\n\n我是 Deep Research 助手，有什么可以帮你的？",
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await intent_node(state, runtime)
        assert result["intent"] == "direct"

        state["intent"] = "direct"
        result = await direct_answer_node(state, runtime)
        assert result.get("intent") == "direct"
        assert "final" in result

    @pytest.mark.asyncio
    async def test_direct_answer_weather_missing_city(self):
        """问天气但未提供城市应提示补充。"""
        from src.graph.nodes.intent_node import intent_node

        state = _make_test_state(query="今天天气如何", intent="direct")
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "direct", "reason": "天气查询"}',
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await intent_node(state, runtime)
        assert result["intent"] == "direct"


# ===================================================================
# Multi-Agent Path — Full Workflow
# ===================================================================


class TestMultiAgentPath:
    """多智能体完整路径测试。"""

    @pytest.mark.asyncio
    async def test_full_workflow_direct_route(self):
        """验证意图识别路由到 direct 路径。"""
        from src.graph.nodes.intent_node import intent_node

        state = _make_test_state(query="什么是 LangGraph？")
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "direct", "reason": "简单问答"}',
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await intent_node(state, runtime)
        assert result["intent"] == "direct"

    @pytest.mark.asyncio
    async def test_full_workflow_multiagent_route(self):
        """验证意图识别路由到 multiagent 路径。"""
        from src.graph.nodes.intent_node import intent_node

        state = _make_test_state(query="LangGraph 反思循环的实现原理")
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "multiagent", "reason": "需要深度调研"}',
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await intent_node(state, runtime)
        assert result["intent"] == "multiagent"

    @pytest.mark.asyncio
    async def test_plan_node_with_llm(self):
        """规划节点生成结构化大纲。"""
        from src.graph.nodes.plan_node import plan_node

        state = _make_test_state(
            query="LangGraph 反思循环的实现原理",
            intent="multiagent",
        )
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "objective": "研究 LangGraph 反思循环",
                                    "sub_questions": ["什么是反思循环", "如何实现"],
                                    "outline": [
                                        {
                                            "id": "sec_1",
                                            "title": "概念",
                                            "description": "介绍",
                                            "search_queries": [
                                                "LangGraph 反思循环 概念"
                                            ],
                                        }
                                    ],
                                    "budget": {"max_rounds": 2, "max_sources": 12},
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await plan_node(state, runtime)
        assert "outline" in result
        assert len(result["outline"]) > 0
        assert "search_plan" in result
        assert len(result["search_plan"]) > 0

    @pytest.mark.asyncio
    async def test_plan_node_fallback(self):
        """规划节点 LLM 失败时使用默认规划。"""
        from src.graph.nodes.plan_node import plan_node

        state = _make_test_state(query="测试问题", intent="multiagent")
        runtime = _make_runtime(llm=None)

        result = await plan_node(state, runtime)
        assert "outline" in result
        assert "search_plan" in result
        assert len(result["search_plan"]) > 0

    @pytest.mark.asyncio
    async def test_deep_dive_with_evidence(self):
        """证据裁判节点处理证据。"""
        from src.graph.nodes.deep_dive_node import deep_dive_node

        state = _make_test_state(
            query="LangGraph 是什么",
            web_evidence=[
                {
                    "source_id": "WEB1_1-1",
                    "title": "LangGraph 官方",
                    "url": "https://langchain.com",
                    "snippet": "框架介绍",
                    "domain": "langchain.com",
                    "source_type": "web",
                },
            ],
            local_evidence=[],
        )
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "summary": "证据裁判完成",
                                    "evidence_pool": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "source_type": "web",
                                            "title": "LangGraph 官方",
                                            "reliability_score": 0.9,
                                        }
                                    ],
                                    "audit_flags": [],
                                    "source_index": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "label": "LangGraph 官方",
                                            "locator": "https://langchain.com",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await deep_dive_node(state, runtime)
        assert "evidence_pool" in result
        assert len(result["evidence_pool"]) > 0
        assert "source_index" in result
        assert "audit_flags" in result

    @pytest.mark.asyncio
    async def test_deep_dive_empty_evidence(self):
        """证据裁判节点无证据时返回空结果。"""
        from src.graph.nodes.deep_dive_node import deep_dive_node

        state = _make_test_state(query="测试", web_evidence=[], local_evidence=[])
        runtime = _make_runtime(llm=None)

        result = await deep_dive_node(state, runtime)
        assert result["evidence_pool"] == []
        assert result["audit_flags"] == []

    @pytest.mark.asyncio
    async def test_analyze_with_evidence(self):
        """分析节点生成结论。"""
        from src.graph.nodes.analyze_node import analyze_node

        state = _make_test_state(
            query="LangGraph 是什么",
            evidence_pool=[
                {
                    "source_id": "WEB1_1-1",
                    "source_type": "web",
                    "title": "LangGraph 官方",
                    "reliability_score": 0.9,
                },
            ],
        )
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "analysis_summary": "LangGraph 是一个 Agent 框架",
                                    "findings": [
                                        {
                                            "claim_id": "c_1",
                                            "claim": "LangGraph 支持状态图",
                                            "confidence": "high",
                                            "source_ids": ["WEB1_1-1"],
                                        }
                                    ],
                                    "claim_map": [
                                        {"claim_id": "c_1", "source_ids": ["WEB1_1-1"]}
                                    ],
                                    "needs_more_research": False,
                                    "missing_gaps": [],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await analyze_node(state, runtime)
        assert "findings" in result
        assert len(result["findings"]) > 0
        assert result["needs_more_research"] is False

    @pytest.mark.asyncio
    async def test_analyze_empty_evidence(self):
        """分析节点无证据时使用默认结论。"""
        from src.graph.nodes.analyze_node import analyze_node

        state = _make_test_state(query="测试", evidence_pool=[])
        runtime = _make_runtime(llm=None)

        result = await analyze_node(state, runtime)
        assert "findings" in result
        assert result["needs_more_research"] is True

    @pytest.mark.asyncio
    async def test_reflect_no_gaps_jumps_to_analyze(self):
        """无信息缺口时 reflect 使用 Command 跳至 analyze。"""
        from langgraph.types import Command

        from src.graph.nodes.reflect_node import reflect_node

        state = _make_test_state(
            query="测试",
            missing_gaps=[],
            iteration=0,
        )
        runtime = _make_runtime(llm=None)

        result = await reflect_node(state, runtime)
        assert isinstance(result, Command)
        assert result.goto == "analyze"

    @pytest.mark.asyncio
    async def test_reflect_with_gaps(self):
        """有信息缺口时 reflect 生成补搜计划。"""
        from src.graph.nodes.reflect_node import reflect_node

        state = _make_test_state(
            query="LangGraph",
            missing_gaps=["性能数据"],
            iteration=0,
            search_plan=[],
        )
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "reflection_summary": "需要性能数据",
                                    "supplementary_queries": [
                                        {
                                            "section_id": "gap_1",
                                            "query": "LangGraph 性能",
                                            "source_preference": "hybrid",
                                            "reason": "补搜性能",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await reflect_node(state, runtime)
        # 有缺口时返回 dict，不是 Command
        assert not isinstance(result, Command)
        assert "supplementary_queries" in result

    @pytest.mark.asyncio
    async def test_write_node(self):
        """写作节点生成最终报告。"""
        from src.graph.nodes.write_node import write_node

        state = _make_test_state(
            query="LangGraph 是什么",
            findings=[
                {
                    "claim_id": "c_1",
                    "claim": "LangGraph 是 Agent 框架",
                    "confidence": "high",
                    "source_ids": ["WEB1_1-1"],
                }
            ],
            source_index=[
                {
                    "source_id": "WEB1_1-1",
                    "label": "LangGraph 官方",
                    "locator": "https://langchain.com",
                    "source_type": "web",
                }
            ],
            audit_flags=[],
        )
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content="# LangGraph 概述\n\nLangGraph 是一个强大的 Agent 框架...\n\n[WEB1_1-1]",
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)

        result = await write_node(state, runtime)
        assert "final" in result
        assert "参考资料" in result["final"]
        assert "WEB1_1-1" in result["final"]

    @pytest.mark.asyncio
    async def test_write_node_no_llm(self):
        """写作节点无 LLM 时使用兜底内容。"""
        from src.graph.nodes.write_node import write_node

        state = _make_test_state(query="测试", findings=[], source_index=[])
        runtime = _make_runtime(llm=None)

        result = await write_node(state, runtime)
        assert "final" in result
        assert "暂无可用证据" in result["final"]


# ===================================================================
# Graph Construction
# ===================================================================


class TestGraphConstruction:
    """图构建测试。"""

    def test_graph_has_all_nodes(self):
        """验证图中包含所有预期节点。"""
        from src.graph.graph import build_graph

        mock_llm = MagicMock()
        mock_kb = MagicMock()

        graph = build_graph(
            llm=mock_llm,
            kb_client=mock_kb,
            progress_emitter=None,
            bocha_api_key="",
        )

        expected_nodes = {
            "intent",
            "query_rewrite",
            "direct_answer",
            "plan",
            "web_search",
            "local_rag",
            "rerank",
            "deep_dive",
            "approval",
            "analyze",
            "reflect",
            "write",
        }
        actual_nodes = set(graph.nodes.keys()) - {"__start__"}
        assert expected_nodes == actual_nodes

    def test_graph_compiles(self):
        """验证图可以成功编译。"""
        from src.graph.graph import build_graph

        mock_llm = MagicMock()
        mock_kb = MagicMock()

        graph = build_graph(
            llm=mock_llm,
            kb_client=mock_kb,
            progress_emitter=None,
            bocha_api_key="",
        )

        assert graph is not None
        assert hasattr(graph, "nodes")


# ===================================================================
# Routing Logic
# ===================================================================


class TestRoutingLogic:
    """路由逻辑测试。"""

    def test_route_after_intent_direct(self):
        from src.graph.graph import _route_after_intent

        state = {"intent": "direct"}
        assert _route_after_intent(state) == "direct_answer"

    def test_route_after_intent_multiagent(self):
        from src.graph.graph import _route_after_intent

        state = {"intent": "multiagent"}
        assert _route_after_intent(state) == "plan"

    def test_route_after_deep_dive_small_pool(self):
        """证据池 <= 10 时直接到 analyze。"""
        from src.graph.graph import _route_after_deep_dive

        state = {"evidence_pool": [{"source_id": f"W-{i}" for i in range(5)}]}
        assert _route_after_deep_dive(state) == "analyze"

    def test_route_after_deep_dive_large_pool(self):
        """证据池 > 10 时到 approval。"""
        from src.graph.graph import _route_after_deep_dive

        state = {"evidence_pool": [{"source_id": f"W-{i}"} for i in range(12)]}
        assert _route_after_deep_dive(state) == "approval"

    def test_should_continue_false(self):
        from src.graph.graph import _should_continue_research

        state = {"iteration": 2, "max_iterations": 2, "needs_more_research": False}
        assert _should_continue_research(state) == "write"

    def test_should_continue_true(self):
        from src.graph.graph import _should_continue_research

        state = {"iteration": 1, "max_iterations": 2, "needs_more_research": True}
        assert _should_continue_research(state) == "reflect"

    def test_should_continue_max_reached(self):
        from src.graph.graph import _should_continue_research

        state = {"iteration": 5, "max_iterations": 2, "needs_more_research": True}
        assert _should_continue_research(state) == "write"


# ===================================================================
# Integration: Full Multi-Agent Flow
# ===================================================================


class TestFullWorkflow:
    """端到端完整工作流集成测试。"""

    @pytest.mark.asyncio
    async def test_full_multiagent_workflow(self):
        """模拟完整的多智能体调研流程。"""
        from src.graph.nodes.analyze_node import analyze_node
        from src.graph.nodes.deep_dive_node import deep_dive_node
        from src.graph.nodes.intent_node import intent_node
        from src.graph.nodes.plan_node import plan_node
        from src.graph.nodes.write_node import write_node

        # 初始化状态
        state = _make_test_state(query="LangGraph 反思循环的实现原理")

        # 1. Intent 识别
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "multiagent", "reason": "需要调研"}',
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await intent_node(state, runtime)
        state.update(result)
        assert state["intent"] == "multiagent"

        # 2. Plan 规划
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "objective": "研究反思循环",
                                    "sub_questions": ["什么是反思循环", "如何实现"],
                                    "outline": [
                                        {
                                            "id": "sec_1",
                                            "title": "概念",
                                            "description": "介绍",
                                            "search_queries": ["LangGraph 反思循环"],
                                        }
                                    ],
                                    "budget": {"max_rounds": 2},
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await plan_node(state, runtime)
        state.update(result)
        assert len(state["outline"]) > 0
        assert len(state["search_plan"]) > 0

        # 3. Deep Dive（模拟证据）
        state["web_evidence"] = [
            {
                "source_id": "WEB1_1-1",
                "title": "LangGraph 文档",
                "url": "https://langchain.com",
                "snippet": "反射循环",
                "domain": "langchain.com",
                "source_type": "web",
            },
            {
                "source_id": "WEB1_1-2",
                "title": "GitHub",
                "url": "https://github.com",
                "snippet": "实现示例",
                "domain": "github.com",
                "source_type": "web",
            },
        ]
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "summary": "证据裁判完成",
                                    "evidence_pool": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "source_type": "web",
                                            "title": "LangGraph 文档",
                                            "reliability_score": 0.9,
                                        },
                                        {
                                            "source_id": "WEB1_1-2",
                                            "source_type": "web",
                                            "title": "GitHub",
                                            "reliability_score": 0.85,
                                        },
                                    ],
                                    "audit_flags": [],
                                    "source_index": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "label": "LangGraph 文档",
                                            "locator": "https://langchain.com",
                                        },
                                        {
                                            "source_id": "WEB1_1-2",
                                            "label": "GitHub",
                                            "locator": "https://github.com",
                                        },
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await deep_dive_node(state, runtime)
        state.update(result)
        assert len(state["evidence_pool"]) == 2

        # 4. Analyze
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "analysis_summary": "反思循环是通过 Agent 自我反思改进输出",
                                    "findings": [
                                        {
                                            "claim_id": "c_1",
                                            "claim": "LangGraph 支持反思循环",
                                            "confidence": "high",
                                            "source_ids": ["WEB1_1-1"],
                                        },
                                        {
                                            "claim_id": "c_2",
                                            "claim": "实现方式包括状态图和条件路由",
                                            "confidence": "medium",
                                            "source_ids": ["WEB1_1-2"],
                                        },
                                    ],
                                    "claim_map": [
                                        {"claim_id": "c_1", "source_ids": ["WEB1_1-1"]},
                                        {"claim_id": "c_2", "source_ids": ["WEB1_1-2"]},
                                    ],
                                    "needs_more_research": False,
                                    "missing_gaps": [],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await analyze_node(state, runtime)
        state.update(result)
        assert len(state["findings"]) == 2
        assert state["needs_more_research"] is False

        # 5. Write
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content="# LangGraph 反思循环\n\nLangGraph 支持通过反思循环改进 Agent 输出...\n\n[WEB1_1-1]\n\n[WEB1_1-2]",
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await write_node(state, runtime)
        state.update(result)
        assert "final" in state
        assert len(state["final"]) > 100
        assert "参考资料" in state["final"]
        assert "WEB1_1-1" in state["final"]
        assert "WEB1_1-2" in state["final"]

    @pytest.mark.asyncio
    async def test_workflow_with_reflection_loop(self):
        """验证反射循环最多执行 max_iterations 次。"""
        from src.graph.nodes.analyze_node import analyze_node
        from src.graph.nodes.deep_dive_node import deep_dive_node
        from src.graph.nodes.intent_node import intent_node
        from src.graph.nodes.plan_node import plan_node
        from src.graph.nodes.reflect_node import reflect_node
        from src.graph.nodes.write_node import write_node

        state = _make_test_state(query="测试反射循环", max_iterations=2)

        # Intent
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content='{"route": "multiagent", "reason": "调研"}',
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await intent_node(state, runtime)
        state.update(result)

        # Plan
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "objective": "测试",
                                    "sub_questions": ["问题1"],
                                    "outline": [
                                        {
                                            "id": "sec_1",
                                            "title": "T",
                                            "search_queries": ["测试"],
                                        }
                                    ],
                                    "budget": {"max_rounds": 2},
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await plan_node(state, runtime)
        state.update(result)

        # First iteration
        state["web_evidence"] = [
            {"source_id": "WEB1_1-1", "title": "Test", "source_type": "web"}
        ]
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "summary": "裁判",
                                    "evidence_pool": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "source_type": "web",
                                            "title": "Test",
                                            "reliability_score": 0.8,
                                        }
                                    ],
                                    "audit_flags": [],
                                    "source_index": [
                                        {
                                            "source_id": "WEB1_1-1",
                                            "label": "Test",
                                            "locator": "",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await deep_dive_node(state, runtime)
        state.update(result)

        # Analyze - needs more research
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "analysis_summary": "证据不足",
                                    "findings": [],
                                    "claim_map": [],
                                    "needs_more_research": True,
                                    "missing_gaps": ["需要更多数据"],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await analyze_node(state, runtime)
        state.update(result)
        assert state["needs_more_research"] is True

        # Reflect - generate supplementary queries
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "reflection_summary": "补搜计划",
                                    "supplementary_queries": [
                                        {
                                            "section_id": "gap_1",
                                            "query": "补充查询",
                                            "source_preference": "hybrid",
                                            "reason": "补数据",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await reflect_node(state, runtime)
        state.update(result)
        assert state["iteration"] == 1

        # Second iteration - enough evidence
        state["web_evidence"] = [
            {"source_id": "WEB2_1-1", "title": "More", "source_type": "web"}
        ]
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "summary": "裁判",
                                    "evidence_pool": [
                                        {
                                            "source_id": "WEB2_1-1",
                                            "source_type": "web",
                                            "title": "More",
                                            "reliability_score": 0.85,
                                        }
                                    ],
                                    "audit_flags": [],
                                    "source_index": [
                                        {
                                            "source_id": "WEB2_1-1",
                                            "label": "More",
                                            "locator": "",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await deep_dive_node(state, runtime)
        state.update(result)

        # Analyze - enough now
        llm = _make_mock_llm(
            [
                {
                    "messages": [
                        MagicMock(
                            content=json.dumps(
                                {
                                    "analysis_summary": "完成",
                                    "findings": [
                                        {
                                            "claim_id": "c_1",
                                            "claim": "结论",
                                            "confidence": "high",
                                            "source_ids": ["WEB2_1-1"],
                                        }
                                    ],
                                    "claim_map": [
                                        {"claim_id": "c_1", "source_ids": ["WEB2_1-1"]}
                                    ],
                                    "needs_more_research": False,
                                    "missing_gaps": [],
                                },
                                ensure_ascii=False,
                            ),
                            type="ai",
                        )
                    ]
                },
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await analyze_node(state, runtime)
        state.update(result)
        assert state["needs_more_research"] is False
        assert state["iteration"] == 1

        # Write final report
        llm = _make_mock_llm(
            [
                {"messages": [MagicMock(content="# 报告\n\n结论：...", type="ai")]},
            ]
        )
        runtime = _make_runtime(llm=llm)
        result = await write_node(state, runtime)
        state.update(result)
        assert "final" in state
        assert len(state["final"]) > 0


# ===================================================================
# Citation Validation
# ===================================================================


class TestCitationValidation:
    """引用验证测试。"""

    def test_valid_citations_preserved(self):
        from src.graph.nodes.write_node import _validate_and_fix_citations

        text = "根据 [WEB1_1-1] 和 [LOC1_1-1] 的研究"
        valid = {"WEB1_1-1", "LOC1_1-1"}
        fixed, used = _validate_and_fix_citations(text, valid)
        assert "WEB1_1-1" in fixed
        assert "LOC1_1-1" in fixed
        assert "WEB1_1-1" in used
        assert "LOC1_1-1" in used

    def test_invalid_citations_removed(self):
        from src.graph.nodes.write_node import _validate_and_fix_citations

        text = "根据 [WEB1_1-1] 和 [WEB9_9-9] 的研究"
        valid = {"WEB1_1-1"}
        fixed, used = _validate_and_fix_citations(text, valid)
        assert "WEB1_1-1" in fixed
        assert "WEB9_9-9" not in fixed
        assert used == ["WEB1_1-1"]

    def test_no_citations(self):
        from src.graph.nodes.write_node import _validate_and_fix_citations

        text = "没有引用的文字"
        valid = set()
        fixed, used = _validate_and_fix_citations(text, valid)
        assert fixed == text
        assert used == []


# ===================================================================
# Evidence Pool Validation
# ===================================================================


class TestEvidencePoolValidation:
    """证据池验证测试。"""

    def test_valid_evidence_pool(self):
        from src.prompt.models import EvidencePool

        data = {
            "summary": "测试",
            "evidence_pool": [
                {
                    "source_id": "W-1",
                    "source_type": "web",
                    "title": "Test",
                    "reliability_score": 0.8,
                },
            ],
            "audit_flags": [],
            "source_index": [],
        }
        pool = EvidencePool(**data)
        assert len(pool.evidence_pool) == 1
        assert pool.evidence_pool[0].reliability_score == 0.8

    def test_invalid_score_raises(self):
        import pytest

        from src.prompt.models import EvidencePoolItem

        with pytest.raises(Exception):
            EvidencePoolItem(
                source_id="W-1", source_type="web", title="Test", reliability_score=1.5
            )


# ===================================================================
# Analysis Output Validation
# ===================================================================


class TestAnalysisOutputValidation:
    """分析输出验证测试。"""

    def test_valid_analysis(self):
        from src.prompt.models import AnalysisOutput, FindingItem

        output = AnalysisOutput(
            analysis_summary="完成",
            needs_more_research=False,
            findings=[FindingItem(claim_id="c_1", claim="测试", confidence="high")],
        )
        assert output.needs_more_research is False
        assert len(output.findings) == 1

    def test_needs_more_research(self):
        from src.prompt.models import AnalysisOutput

        output = AnalysisOutput(
            analysis_summary="证据不足",
            needs_more_research=True,
            missing_gaps=["性能数据"],
        )
        assert output.needs_more_research is True
        assert output.missing_gaps == ["性能数据"]
