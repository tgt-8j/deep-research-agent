"""重构后的工作流服务。

替代原来的 app/backend/service/workflow_service.py，
使用新的 src/ 模块结构。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager

try:  # LangGraph >= 1.2：节点里抛出的 CancelledError 会被包成 NodeCancelledError
    from langgraph.errors import NodeCancelledError
except ImportError:  # 1.0.x / 1.1.x 没有这个异常，原生 CancelledError 会直接冒泡

    class NodeCancelledError(Exception):  # noqa: N818
        """旧版 LangGraph 的占位异常：永远不会被抛出，只为让 except 子句可用。"""


from ..config import AppConfig
from ..graph import build_graph
from ..llm import create_default_llm
from ..persistence import SessionStore, WorkflowCancellation, cancellation_scope
from ..retrieval import create_knowledge_base_client
from ..state import ResearchState

logger = logging.getLogger("research.services.workflow")


@contextmanager
def _translate_node_cancellation():
    """把 LangGraph 的 ``NodeCancelledError`` 统一成 ``asyncio.CancelledError``。

    LangGraph 会把节点里抛出的 CancelledError 包成 ``NodeCancelledError``，
    而它继承自 Exception —— 服务层/路由层的 ``except Exception`` 会把它当成
    普通错误上报，前端就看不到"已取消"。在服务边界翻译回来，保证取消对外
    只有一个语义。
    """
    try:
        yield
    except NodeCancelledError as exc:
        raise asyncio.CancelledError(str(exc)) from exc


class WorkflowService:
    """重构后的工作流服务。

    核心改进：
    1. 不再使用全局单例，改为在构建时传入依赖
    2. 支持 SSE 流式进度推送
    3. 使用新的 State 和 Context 模式
    4. 支持取消信号（来自 Paper-Agent 模式）
    5. 支持会话持久化（来自 Paper-Agent 模式）
    """

    def __init__(self, config: AppConfig, session_store: SessionStore | None = None):
        self._config = config
        self._session_store = session_store
        self._llm = None
        self._kb_client = None
        self._graph = None

    def _ensure_initialized(self) -> None:
        """延迟初始化：首次使用时才创建依赖。"""
        if self._graph is not None:
            return

        # 创建 LLM 适配器
        self._llm = create_default_llm()

        # 创建知识库客户端
        self._kb_client = create_knowledge_base_client(
            milvus_host=self._config.milvus_host,
            milvus_port=self._config.milvus_port,
            collection_name=self._config.milvus_collection,
            enable_milvus=self._config.enable_milvus,
            api_key=self._config.api_key,
        )

        # 构建图（传入取消信号）
        self._graph = build_graph(
            llm=self._llm,
            kb_client=self._kb_client,
            progress_emitter=None,  # SSE 由上层注入
            bocha_api_key=os.getenv(
                "BOCHA_API_KEY", ""
            ),  # 从环境变量 BOCHA_API_KEY 读取
        )
        logger.info("工作流服务初始化完成")

    def _build_state(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
    ) -> ResearchState:
        """构建初始状态。

        注意：取消信号**不**放进 state。它含 asyncio.Event、不可序列化，
        塞进 state 会污染 checkpointer 快照；改为运行时用
        ``cancellation_scope`` 绑定到上下文（见 graph.py 的取消边界）。
        """
        return {
            "query": query,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "memory_context": "",
            "messages": [],
            "intent": "",
            "phase": "initialized",
            "rewritten_queries": [],
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
            "iteration": 0,
            "max_iterations": self._config.max_iterations,
        }

    async def run(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        cancellation: WorkflowCancellation | None = None,
    ) -> str:
        """执行调研任务，返回最终报告。"""
        self._ensure_initialized()
        state = self._build_state(query, user_id, thread_id, tenant_id)
        config = {"configurable": {"thread_id": thread_id}}

        with _translate_node_cancellation(), cancellation_scope(cancellation):
            result = await self._graph.ainvoke(state, config)
        return result.get("final", "")

    async def resume(
        self,
        thread_id: str,
        checkpoint_state: dict,
        cancellation: WorkflowCancellation | None = None,
    ) -> str:
        """从 Checkpoint 恢复中断的调研任务。

        Args:
            thread_id: 线程 ID（与原始请求一致）
            checkpoint_state: 保存的 state 快照（来自 SessionStore.Checkpoint）
            cancellation: 取消信号

        Returns:
            最终报告文本
        """
        self._ensure_initialized()
        # 使用相同的 thread_id，LangGraph checkpointer 会自动恢复中间状态
        config = {"configurable": {"thread_id": thread_id}}

        # 合并快照中的业务数据
        restored_state = dict(checkpoint_state)

        with _translate_node_cancellation():
            try:
                # 尝试从 checkpointer 恢复；如果 checkpoint 不存在则从头开始
                config = {"configurable": {"thread_id": thread_id}}
                with cancellation_scope(cancellation):
                    result = await self._graph.ainvoke(restored_state, config=config)
            except Exception:
                # Checkpoint 可能不在 checkpointer 中（进程重启后），降级为从头执行
                logger.warning(
                    "Checkpoint 恢复失败，降级为从头执行 | thread_id=%s", thread_id
                )
                config = {"configurable": {"thread_id": thread_id}}
                with cancellation_scope(cancellation):
                    result = await self._graph.ainvoke(restored_state, config=config)

        return result.get("final", "")

    async def _iterate_events(
        self,
        state: ResearchState,
        config: dict,
        cancellation: WorkflowCancellation | None,
    ) -> AsyncIterator[dict]:
        """在取消信号的上下文中迭代图事件。

        ContextVar 只在图执行期间绑定：asyncio 每个任务持有独立的 context
        副本，并发请求互不串扰；``with`` 退出时自动恢复，不会泄漏给后续任务。
        """
        with _translate_node_cancellation(), cancellation_scope(cancellation):
            async for event in self._graph.astream_events(state, config, version="v2"):
                yield event
            # LangGraph 1.0.x 的 astream_events 会把节点里抛出的 CancelledError
            # 静默吞掉（流直接枯竭、不抛异常）；1.2+ 才会包成 NodeCancelledError
            # 冒泡。这里在流正常结束后按取消信号补一次判断，两个版本语义一致。
            if cancellation is not None and cancellation.is_requested():
                raise asyncio.CancelledError()

    async def stream_events(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        emit: Callable[[dict], None],
        cancellation: WorkflowCancellation | None = None,
    ) -> None:
        """执行调研并实时推送进度事件（用于 SSE）。

        参考 Paper-Agent 的 stream_aggregator 模式：
        - 每个节点开始/结束时推送进度
        - LLM 流式输出实时推送
        - 最终结果推送
        """
        self._ensure_initialized()
        state = self._build_state(query, user_id, thread_id, tenant_id)
        config = {"configurable": {"thread_id": thread_id}}

        emit({"type": "status", "message": "任务已接收，正在初始化工作流"})

        # 使用 stream 模式逐个获取节点输出，同时收集最终结果
        final = ""
        intent = "multiagent"
        async for event in self._iterate_events(state, config, cancellation):
            kind = event.get("event")
            if kind == "on_chain_start" and "name" in event:
                node_name = event["name"]
                emit(
                    {
                        "type": "progress",
                        "node": node_name,
                        "message": f"{node_name} 节点开始执行",
                        "status": "running",
                    }
                )
            elif kind == "on_chain_end" and "name" in event:
                node_name = event["name"]
                emit(
                    {
                        "type": "progress",
                        "node": node_name,
                        "message": f"{node_name} 节点执行完成",
                        "status": "success",
                    }
                )
                # 从节点输出中收集 final 和 intent
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    if "final" in output and not final:
                        final = output.get("final", "")
                    if "intent" in output:
                        intent = output.get("intent", "multiagent")
            elif kind == "on_chat_model_stream":
                # data.chunk 在真实运行时是 AIMessageChunk 对象（有 .content 属性、
                # 没有 .get）；单测里 mock 成 dict。两种都要兼容，否则流式一进来就抛
                # "'AIMessageChunk' object has no attribute 'get'"，整条 SSE 中断。
                chunk = (event.get("data") or {}).get("chunk")
                if isinstance(chunk, dict):
                    content = chunk.get("content", "")
                else:
                    content = getattr(chunk, "content", "")
                if content:
                    emit(
                        {
                            "type": "delta",
                            "content": str(content),
                        }
                    )
            elif kind == "on_checkpoint":
                # 记录 checkpoint 事件，用于恢复中断流程
                pass

        # 如果流中没有拿到 final，兜底调用一次 ainvoke
        if not final:
            with _translate_node_cancellation(), cancellation_scope(cancellation):
                result = await self._graph.ainvoke(state, config)
            final = result.get("final", "")
            intent = result.get("intent", "multiagent")

        # 持久化会话
        if self._session_store:
            self._session_store.update_session_status(
                session_key=thread_id,
                status="completed",
                final_result=final[:200],
            )

        emit(
            {
                "type": "route",
                "message": "已走直接回答路径"
                if intent == "direct"
                else "已走多智能体研究路径",
            }
        )
        emit(
            {
                "type": "final",
                "query": query,
                "user_id": user_id,
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "final": final,
            }
        )
