"""节点模块入口：统一导出所有 LangGraph 工作流节点函数。

每个节点文件对应一个图节点，遵循 shopkeeper-agent 的单职责原则：
- 一个文件只做一件事
- 节点签名统一为 (state, runtime) -> dict
- 通过 runtime.context 获取外部依赖，不直接使用全局单例
"""

from .intent_node import intent_node
from .query_rewrite_node import query_rewrite_node
from .plan_node import plan_node
from .web_search_node import web_search_node
from .local_rag_node import local_rag_node
from .rerank_node import rerank_node
from .deep_dive_node import deep_dive_node
from .approval_node import approval_node
from .analyze_node import analyze_node
from .reflect_node import reflect_node
from .write_node import write_node
from .direct_answer_node import direct_answer_node

__all__ = [
    "intent_node",
    "query_rewrite_node",
    "plan_node",
    "web_search_node",
    "local_rag_node",
    "rerank_node",
    "deep_dive_node",
    "approval_node",
    "analyze_node",
    "reflect_node",
    "write_node",
    "direct_answer_node",
]
