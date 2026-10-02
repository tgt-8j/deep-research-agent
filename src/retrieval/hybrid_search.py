"""混合检索模块：BM25 关键词检索 + Embedding 向量检索的 RRF 融合。

设计思路：
- 纯向量检索对专有名词（如 "LangGraph"）召回差
- BM25 擅长关键词精确匹配，但缺乏语义泛化能力
- 两者用 Reciprocal Rank Fusion (RRF) 融合，兼顾精度和召回
- RRF 公式：score = Σ(1 / (k + rank_i))，k 通常取 60

使用方式：
    from src.retrieval.hybrid_search import hybrid_search, rrf_merge
    results = hybrid_search(query, documents, k_vector=10, k_bm25=10, rrf_k=60)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("research.retrieval.hybrid")

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    logger.warning("[hybrid_search] rank-bm25 未安装，仅使用向量检索")


def _tokenize_chinese(text: str) -> list[str]:
    """中文分词：按字符切分 + 保留英文单词。"""
    import re
    # 英文单词和数字
    words = re.findall(r"[a-zA-Z0-9_]+", text)
    # 中文字符（单个字作为 token）
    chars = re.findall(r"[一-鿿]", text)
    return words + chars


def _tokenize(text: str) -> list[str]:
    """简单分词：英文按空白切分，中文按字符切分。"""
    if not text.strip():
        return []
    # 检查是否包含中文
    has_chinese = any("一" <= c <= "鿿" for c in text)
    if has_chinese:
        return _tokenize_chinese(text)
    return text.lower().split()


def rrf_merge(
    ranked_results: list[list[dict]],
    rrf_k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion 融合多个排序结果。

    Args:
        ranked_results: 多个排序列表，每个元素是 {"doc_id": str, ...} 的 dict
        rrf_k: RRF 平滑参数，默认 60

    Returns:
        融合后的排序列表，按 RRF score 降序
    """
    # 计算每个文档的 RRF 分数
    scores: dict[str, float] = {}
    for ranking in ranked_results:
        for rank, doc in enumerate(ranking, 1):
            doc_id = doc.get("doc_id") or doc.get("source_id") or str(doc)
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (rrf_k + rank)

    # 按分数降序排序
    all_docs: dict[str, dict] = {}
    for ranking in ranked_results:
        for doc in ranking:
            doc_id = doc.get("doc_id") or doc.get("source_id") or str(doc)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

    merged = []
    for doc_id, doc in all_docs.items():
        score = scores.get(doc_id, 0)
        if score > 0:
            merged.append({"_doc_id": doc_id, **doc, "_rrf_score": score})
    merged.sort(key=lambda x: x["_rrf_score"], reverse=True)
    return merged


def bm25_search(
    query: str,
    documents: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """BM25 关键词检索。

    Args:
        query: 查询词
        documents: 文档列表，每个元素是 {"doc_id": str, "content": str, ...}
        top_k: 返回数量

    Returns:
        按 BM25 分数排序的结果列表
    """
    if not _HAS_BM25 or not documents:
        return []

    # 构建 corpus（每个文档的分词结果）
    corpus = [_tokenize(doc.get("content", "") or doc.get("snippet", "")) for doc in documents]
    query_tokens = _tokenize(query)

    if not query_tokens or not any(len(tokens) > 0 for tokens in corpus):
        return []

    try:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
    except Exception as e:
        logger.warning("[bm25_search] BM25 计算失败: %s", e)
        return []

    # 按分数排序
    doc_scores = list(enumerate(scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in doc_scores[:top_k]:
        if score > 0:
            doc = dict(documents[idx])
            doc["_bm25_score"] = score
            results.append(doc)

    return results


def hybrid_search(
    query: str,
    documents: list[dict],
    embeddings: Any = None,
    k_vector: int = 10,
    k_bm25: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """混合检索：BM25 + 向量检索 RRF 融合。

    Args:
        query: 查询词
        documents: 文档列表，每个元素需要包含 doc_id/source_id 和 content/snippet
        embeddings: Embedding 函数（可选），接收 query 返回向量
        k_vector: 向量检索返回数量
        k_bm25: BM25 检索返回数量
        rrf_k: RRF 平滑参数

    Returns:
        融合后的排序列表
    """
    if not documents:
        return []

    # 确保文档有 doc_id
    for doc in documents:
        if "doc_id" not in doc and "source_id" not in doc:
            doc["doc_id"] = doc.get("title", str(doc))[:50]

    # BM25 检索
    bm25_results = bm25_search(query, documents, top_k=k_bm25)
    logger.info("[hybrid_search] BM25 返回 %d 条", len(bm25_results))

    # 向量检索（如果有 embeddings）
    vector_results: list[dict] = []
    if embeddings is not None:
        try:
            query_vector = embeddings(query)
            # 假设 embeddings 返回向量，需要调用 vectorstore.similarity_search_with_score
            # 这里简化处理，实际项目中需要传入 vectorstore
            logger.info("[hybrid_search] 向量检索暂不支持，仅使用 BM25")
        except Exception as e:
            logger.warning("[hybrid_search] 向量检索失败: %s", e)

    # RRF 融合
    all_rankings = [bm25_results] + ([vector_results] if vector_results else [])
    if len(all_rankings) == 1:
        # 只有 BM25，直接返回
        for doc in bm25_results:
            doc.pop("_bm25_score", None)
        return bm25_results

    merged = rrf_merge(all_rankings, rrf_k=rrf_k)

    # 清理内部字段
    for doc in merged:
        doc.pop("_rrf_score", None)
        doc.pop("_bm25_score", None)

    return merged


__all__ = ["hybrid_search", "rrf_merge", "bm25_search", "_tokenize"]
