"""Runtime Context：为 Agent 节点提供共享的外部依赖，避免全局单例。

设计模式（参考 shopkeeper-agent 的 DataAgentContext）：
- 节点通过 runtime.context 获取依赖，不直接使用全局 init 函数
- 外部依赖（RAG、Web Search API Key 等）在图编译时一次性传入
- State 中只保留业务数据，Context 中只保留基础设施
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict


# 检索来源偏好
SourcePreference = Literal["web", "local", "hybrid"]


class RetrievalConfig(TypedDict, total=False):
    """与检索相关的配置参数。"""

    bocha_api_key: str
    milvus_enabled: bool
    max_web_results: int
    max_local_results: int


class WorkflowProgressEmitter(TypedDict, total=False):
    """流式进度上报接口。

    节点调用此函数将执行进度推送给前端，用于 SSE 实时展示。
    """

    def __call__(
        self,
        event_type: str,
        *,
        step: str,
        status: Literal["running", "success", "error"] = "running",
        **extra,
    ) -> None:
        ...


class ResearchRuntimeContext(TypedDict, total=False):
    """LangGraph Runtime 中传递的业务上下文。

    对应 shopkeeper-agent 的 DataAgentContext，但适配本项目需求：
    - 保留 Bocha Web Search API Key（无需持久连接，每次调用时传 key）
    - 保留 Milvus 启用标志（用于判断本地检索是否可用）
    - 保留进度上报回调（供各节点推送实时状态给前端）
    """

    retrieval_config: RetrievalConfig
    progress_emitter: WorkflowProgressEmitter


class SearchClient:
    """统一检索客户端接口。

    封装网页搜索和本地知识库搜索，节点只需依赖此接口而不直接导入具体实现。
    """

    def search_web(self, query: str, count: int = 4) -> list[dict]:
        """执行网页搜索，返回原始记录列表。"""
        raise NotImplementedError

    def search_local(self, query: str, limit: int = 4) -> list[dict]:
        """执行本地知识库检索，返回原始记录列表。"""
        raise NotImplementedError
