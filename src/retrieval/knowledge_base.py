"""本地知识库检索封装。

从原来的 global _rag_system_instance 单例模式改为显式传入 RAGSystem 实例，
便于测试和依赖注入。
"""

from __future__ import annotations

import logging

# 直接从原模块导入，保持兼容性
import sys
from pathlib import Path

_app_root = Path(__file__).resolve().parents[1]
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))
from app.mult_agents.rag.core import RAGConfig, RAGSystem  # noqa: F401  # 供外部引用

from .hybrid_search import hybrid_search

logger = logging.getLogger(__name__)


class KnowledgeBaseClient:
    """本地知识库检索客户端（包装 RAGSystem）。

    设计原则：
    - 不持有全局状态，每次调用显式传入 rag_system
    - 提供 with_context 作为临时注入点的便捷方法
    - 支持混合检索（BM25 + 向量 RRF 融合）
    """

    def __init__(self, rag_system: RAGSystem | None = None):
        self._rag_system = rag_system

    def search(self, query: str, limit: int = 4) -> list[dict]:
        """执行本地知识库检索。"""
        if self._rag_system is None:
            return []
        try:
            return self._rag_system.search_records(query, k=limit)
        except Exception:
            return []

    def search_text(self, query: str) -> str:
        """执行检索并返回文本摘要（兼容旧接口）。"""
        if self._rag_system is None:
            return "RAG 系统未初始化"
        try:
            return self._rag_system.search(query)
        except Exception:
            return ""

    def hybrid_search(self, query: str, limit: int = 4) -> list[dict]:
        """混合检索：BM25 关键词 + 向量语义 RRF 融合。

        Args:
            query: 查询词
            limit: 返回数量

        Returns:
            融合后的检索结果列表
        """
        if self._rag_system is None:
            return []
        try:
            # 获取所有文档用于 BM25
            all_docs = self._rag_system.search_records(query, k=100)
            if not all_docs:
                return []

            # 构建 BM25 需要的格式
            bm25_docs = []
            for doc in all_docs:
                bm25_docs.append(
                    {
                        "doc_id": doc.get("doc_id") or doc.get("source_id", ""),
                        "content": doc.get("snippet", "")
                        or doc.get("page_content", ""),
                        **doc,
                    }
                )

            # 执行混合检索
            results = hybrid_search(
                query=query,
                documents=bm25_docs,
                k_vector=limit,
                k_bm25=limit,
                rrf_k=60,
            )

            # 清理内部字段，返回标准格式
            clean_results = []
            for r in results[:limit]:
                clean_results.append(
                    {
                        "source_id": r.get("source_id", r.get("doc_id", "")),
                        "doc_id": r.get("doc_id", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", r.get("content", ""))[:500],
                        "source_type": "local",
                        "rrf_score": r.get("_rrf_score", 0),
                    }
                )
            return clean_results
        except Exception as e:
            logger.warning("[hybrid_search] 混合检索失败，降级为普通检索: %s", e)
            return self.search(query, limit)


def create_knowledge_base_client(
    milvus_host: str,
    milvus_port: int,
    collection_name: str,
    enable_milvus: bool,
    api_key: str,
) -> KnowledgeBaseClient:
    """根据配置创建 KnowledgeBaseClient 实例。

    替代原来的 init_rag_system() 全局函数，改为工厂函数返回实例。
    """
    from ..mult_agents.rag.core import RAGConfig

    config = RAGConfig(
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        enable_milvus=enable_milvus,
    )
    try:
        rag_system = RAGSystem(api_key, config)
        return KnowledgeBaseClient(rag_system=rag_system)
    except Exception as e:
        print(f"知识库客户端初始化失败: {e}")
        return KnowledgeBaseClient(rag_system=None)
