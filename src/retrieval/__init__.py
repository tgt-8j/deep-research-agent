"""检索模块入口。"""

from .knowledge_base import KnowledgeBaseClient, create_knowledge_base_client
from .reranker import Reranker, rerank_evidence
from .web_search import (
    SimpleRateLimiter,
    _estimate_relevance,
    _extract_query_terms,
    _filter_records,
    _is_official_domain,
    bocha_web_search,
    create_rate_limiter,
    extract_url_content,
    get_rate_limiter,
)

__all__ = [
    "KnowledgeBaseClient",
    "Reranker",
    "SimpleRateLimiter",
    "_estimate_relevance",
    "_extract_query_terms",
    "_filter_records",
    "_is_official_domain",
    "bocha_web_search",
    "create_knowledge_base_client",
    "create_rate_limiter",
    "extract_url_content",
    "get_rate_limiter",
    "rerank_evidence",
]
