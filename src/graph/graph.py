"""工作流编排模块：定义 LangGraph 节点、条件路由与整体执行路径。

参考 shopkeeper-agent 的 graph.py 和 Paper-Agent 的 graph.py：
- 使用 StateGraph 定义状态机
- 条件边实现分支逻辑
- 支持子循环（reflect -> web_search/local_rag）
- 【Paper-Agent 借鉴】每个节点用取消边界包裹，支持用户中途停止
- 【Paper-Agent 借鉴】使用 MemorySaver checkpointer 支持中断恢复
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from .nodes import (
    intent_node,
    query_rewrite_node,
    direct_answer_node,
    plan_node,
    web_search_node,
    local_rag_node,
    rerank_node,
    deep_dive_node,
    approval_node,
    analyze_node,
    reflect_node,
    write_node,
)
from ..state import ResearchState

logger = logging.getLogger("research.graph")


def _route_after_intent(state: ResearchState) -> str:
    """根据意图决定走直接回答还是深度调研。"""
    if state.get("intent") == "direct":
        return "direct_answer"
    return "plan"


def _route_after_deep_dive(state: ResearchState) -> str:
    """根据证据数量决定是走人工审批还是直接分析。"""
    evidence_count = len(state.get("evidence_pool", []))
    if evidence_count > 10:
        return "approval"
    return "analyze"


def _should_continue_research(state: ResearchState) -> str:
    """判断是否需要继续反思补搜，还是结束并写作。"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if iteration >= max_iter:
        return "write"
    if state.get("needs_more_research", False):
        return "reflect"
    return "write"


def _with_cancellation_boundary(node_name: str, node: Callable) -> Callable:
    """【借鉴 Paper-Agent】给节点加上取消检查边界。

    节点开始前和结束后各检查一次 cancellation 信号。
    如果用户点了停止，丢弃当前节点的未提交结果，
    让恢复时从上一个稳定状态重新执行。
    """

    async def _guarded(state: ResearchState, **kwargs) -> dict:
        # 从 runtime context 获取取消信号
        runtime = kwargs.get("runtime")
        cancellation = getattr(runtime, "cancellation", None) if runtime else None

        # 开始检查
        if cancellation is not None:
            cancellation.raise_if_requested()

        result = await node(state, **kwargs)

        # 结束检查（确保节点中间结果不污染状态）
        if cancellation is not None:
            cancellation.raise_if_requested()

        return result

    _guarded.__name__ = f"{node_name}_guarded"
    return _guarded


def build_graph(
    llm: Any,
    kb_client: Any,
    progress_emitter: Callable | None = None,
    bocha_api_key: str = "",
    cancellation: Any = None,  # WorkflowCancellation 实例
) -> StateGraph:
    """构建深度调研工作流图。

    Args:
        llm: LLMAdapter 实例
        kb_client: KnowledgeBaseClient 实例
        progress_emitter: SSE 进度上报回调
        bocha_api_key: Bocha Web Search API Key
        cancellation: 取消信号对象（Paper-Agent 模式）

    Returns:
        编译后的 StateGraph（带 MemorySaver checkpointer）
    """
    workflow = StateGraph(ResearchState)

    # 定义节点的包装函数，注入 Runtime Context + 取消边界
    def _make_node(node_fn: Callable) -> Callable:
        def wrapped(state: ResearchState, **kwargs) -> dict:
            from ..context import ResearchRuntimeContext
            context: ResearchRuntimeContext = {
                "llm": llm,
                "kb_client": kb_client,
                "progress_emitter": progress_emitter,
                "retrieval_config": {"bocha_api_key": bocha_api_key},
                "cancellation": cancellation,  # 传递取消信号
            }
            # 将 context 注入到 kwargs 中，节点通过 runtime.context 访问
            class Runtime:
                class context:
                    llm = context["llm"]
                    kb_client = context["kb_client"]
                    progress_emitter = context["progress_emitter"]
                    retrieval_config = context["retrieval_config"]
                    cancellation = context["cancellation"]
            return node_fn(state, Runtime)
        return wrapped

    # 添加节点（每个节点都包裹取消边界）
    workflow.add_node("intent", _with_cancellation_boundary("intent", _make_node(intent_node)))
    workflow.add_node("query_rewrite", _with_cancellation_boundary("query_rewrite", _make_node(query_rewrite_node)))
    workflow.add_node("direct_answer", _with_cancellation_boundary("direct_answer", _make_node(direct_answer_node)))
    workflow.add_node("plan", _with_cancellation_boundary("plan", _make_node(plan_node)))
    workflow.add_node("web_search", _with_cancellation_boundary("web_search", _make_node(web_search_node)))
    workflow.add_node("local_rag", _with_cancellation_boundary("local_rag", _make_node(local_rag_node)))
    workflow.add_node("rerank", _with_cancellation_boundary("rerank", _make_node(rerank_node)))
    workflow.add_node("deep_dive", _with_cancellation_boundary("deep_dive", _make_node(deep_dive_node)))
    workflow.add_node("approval", _with_cancellation_boundary("approval", _make_node(approval_node)))
    workflow.add_node("analyze", _with_cancellation_boundary("analyze", _make_node(analyze_node)))
    workflow.add_node("reflect", _with_cancellation_boundary("reflect", _make_node(reflect_node)))
    workflow.add_node("write", _with_cancellation_boundary("write", _make_node(write_node)))

    # 定义边
    workflow.add_edge(START, "intent")
    workflow.add_edge("intent", "query_rewrite")
    workflow.add_conditional_edges(
        "query_rewrite",
        _route_after_intent,
        {"direct_answer": "direct_answer", "plan": "plan"},
    )
    workflow.add_edge("plan", "web_search")
    workflow.add_edge("plan", "local_rag")
    workflow.add_edge("web_search", "rerank")
    workflow.add_edge("local_rag", "rerank")
    workflow.add_edge("rerank", "deep_dive")
    workflow.add_conditional_edges(
        "deep_dive",
        _route_after_deep_dive,
        {"approval": "approval", "analyze": "analyze"},
    )
    workflow.add_edge("approval", "analyze")

    workflow.add_conditional_edges(
        "analyze",
        _should_continue_research,
        {"reflect": "reflect", "write": "write"},
    )
    workflow.add_edge("reflect", "web_search")
    workflow.add_edge("reflect", "local_rag")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("write", END)

    # 使用 MemorySaver checkpointer 支持中断恢复和会话持久化
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
