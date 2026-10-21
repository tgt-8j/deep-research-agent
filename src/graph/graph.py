"""工作流编排模块：定义 LangGraph 节点、条件路由与整体执行路径。

- 使用 StateGraph 定义状态机，条件边实现分支逻辑
- 支持子循环（reflect -> web_search/local_rag）
- 每个节点用取消边界包裹，支持用户中途停止
- 使用 MemorySaver checkpointer 支持中断恢复

取消信号的传递方式见 ``src/persistence/cancellation.py``：
图是编译一次、多请求复用的，所以信号走 ContextVar 而不是节点闭包。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..persistence.cancellation import WorkflowCancellation, current_cancellation
from ..state import ResearchState
from .nodes import (
    analyze_node,
    approval_node,
    deep_dive_node,
    direct_answer_node,
    intent_node,
    local_rag_node,
    plan_node,
    query_rewrite_node,
    reflect_node,
    rerank_node,
    web_search_node,
    write_node,
)

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


def _raise_if_cancelled(
    cancellation: WorkflowCancellation | None,
    node_name: str,
    phase: str,
) -> None:
    """在节点边界检查停止请求；命中则抛 ``CancelledError`` 中止本次运行。"""
    if cancellation is None or not cancellation.is_requested():
        return
    logger.info("[cancel] %s 节点（%s）检测到停止请求，中止本次运行", node_name, phase)
    cancellation.raise_if_requested()


def _with_cancellation_boundary(
    node_name: str,
    node: Callable,
    default_cancellation: WorkflowCancellation | None = None,
) -> Callable:
    """给节点加上取消检查边界。

    取消信号优先取当前异步上下文绑定的那个（``current_cancellation()``），
    没绑定时退回 ``build_graph`` 传入的默认值。请求级信号由
    ``cancellation_scope`` 在服务层绑定 —— 图编译一次、多请求复用，
    信号不能挂在闭包上，否则并发请求会互相取消。

    节点开始前和结束后各检查一次：结束后命中就丢弃该节点的返回值，
    让恢复时从上一个稳定 checkpoint 重新执行。
    """

    async def _guarded(state: ResearchState, **kwargs) -> dict:
        cancellation = current_cancellation() or default_cancellation

        _raise_if_cancelled(cancellation, node_name, "开始前")

        result = node(state, **kwargs)
        if inspect.isawaitable(result):
            result = await result

        # 节点执行期间可能刚收到停止请求：丢弃未提交的结果
        _raise_if_cancelled(cancellation, node_name, "结束后")

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
        cancellation: 默认取消信号；请求级信号由 ``cancellation_scope`` 绑定后优先生效

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
                # 请求级信号来自 ContextVar；build_graph 的入参只是默认值
                "cancellation": current_cancellation() or cancellation,
            }

            # 将 context 注入到 kwargs 中，节点通过 runtime.context 访问
            class Runtime:
                def __init__(self, ctx):
                    self.context = ctx

                @property
                def llm(self):
                    return self.context["llm"]

                @property
                def kb_client(self):
                    return self.context["kb_client"]

                @property
                def progress_emitter(self):
                    return self.context["progress_emitter"]

                @property
                def retrieval_config(self):
                    return self.context["retrieval_config"]

                @property
                def cancellation(self):
                    return self.context["cancellation"]

            return node_fn(state, Runtime(context))

        return wrapped

    def _guard(name: str, node_fn: Callable) -> Callable:
        """节点包装：注入 Runtime Context + 取消边界。"""
        return _with_cancellation_boundary(name, _make_node(node_fn), cancellation)

    # 添加节点（每个节点都包裹取消边界）
    workflow.add_node("intent", _guard("intent", intent_node))
    workflow.add_node("query_rewrite", _guard("query_rewrite", query_rewrite_node))
    workflow.add_node("direct_answer", _guard("direct_answer", direct_answer_node))
    workflow.add_node("plan", _guard("plan", plan_node))
    workflow.add_node("web_search", _guard("web_search", web_search_node))
    workflow.add_node("local_rag", _guard("local_rag", local_rag_node))
    workflow.add_node("rerank", _guard("rerank", rerank_node))
    workflow.add_node("deep_dive", _guard("deep_dive", deep_dive_node))
    workflow.add_node("approval", _guard("approval", approval_node))
    workflow.add_node("analyze", _guard("analyze", analyze_node))
    workflow.add_node("reflect", _guard("reflect", reflect_node))
    workflow.add_node("write", _guard("write", write_node))

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
