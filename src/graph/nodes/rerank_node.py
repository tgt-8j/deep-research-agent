"""Rerank 重排序节点：对 web/local 两路检索结果进行粗筛+精排。

职责：
- 合并 web_evidence + local_evidence
- 对 web/local 两路做 RRF 双路融合（按 source_id 去重，两路都出现的证据分数更高）
- 用 Embedding 向量相似度做两级重排序（粗筛 Top-30，精排 Top-10）
- 输出 rerank_stats 统计信息和重新排序后的 evidence_pool
- 降级策略：Embedding 不可用时跳过重排序，直接用 RRF 融合结果作为输出

设计原则：
- 不修改原始证据数据，只改变排序
- rerank_stats 记录重排序过程，便于调试和面试讲解
- 与 deep_dive_node 分离：rerank 只做排序，deep_dive 做评分/去重/冲突检测
"""

from __future__ import annotations

import logging
from typing import Any

from app.metrics import track_node

from ...retrieval.hybrid_search import rrf_merge
from ...state import ResearchState

logger = logging.getLogger("research.nodes.rerank")


def _rrf_fuse_two_sources(
    web_evidence: list[dict],
    local_evidence: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """对 web 和 local 两路证据做 RRF 融合。

    两路各视为一个独立 ranking：
    - 在 web 路出现 → 得分 1 / (k + web_rank)
    - 在 local 路出现 → 得分 1 / (k + local_rank)
    - 两路都出现（source_id 相同）→ 两项相加，自动排到最前

    返回融合后的证据列表（含 _rrf_fuse_score 字段），按分数降序。
    """

    # 以 url（web 路）或 doc_id（local 路）为主键，source_id 兜底
    # 同一 URL 在多个 query 里被搜到时，RRF 可以正确合并；
    # 跨源（web URL vs local doc_id）去重需要上游统一 source 标识，当前数据模型暂不支持
    def _key(item: dict) -> str:
        url = item.get("url", "")
        if url:
            return f"url:{url}"
        doc_id = item.get("doc_id", "")
        if doc_id:
            return f"doc:{doc_id}"
        sid = item.get("source_id", "")
        if sid:
            return f"sid:{sid}"
        return f"title:{item.get('title', '')}_{item.get('snippet', '')[:50]}"

    # 两路各自保持原始顺序（已经是各路检索器的排序结果）
    all_rankings: list[list[dict]] = []
    if web_evidence:
        all_rankings.append(web_evidence)
    if local_evidence:
        all_rankings.append(local_evidence)

    if not all_rankings:
        return []

    if len(all_rankings) == 1:
        # 只有一路有结果，直接透传
        fused = [dict(item) for item in all_rankings[0]]
        for item in fused:
            item["_rrf_fuse_score"] = 0.0
            item["_fuse_source"] = "single_source"
        return fused

    # 两路都有结果，做 RRF 融合
    # rrf_merge 用 doc.get("doc_id") 取 key，把 _key() 结果临时放进 doc_id 字段
    # 融合后从 _doc_id（rrf_merge 注入）反解析原始 doc_id 值，避免丢失
    web_keyed = [{**item, "doc_id": _key(item)} for item in web_evidence]
    local_keyed = [{**item, "doc_id": _key(item)} for item in local_evidence]
    fused_raw = rrf_merge([web_keyed, local_keyed], rrf_k=rrf_k)

    fused: list[dict] = []
    for item in fused_raw:
        # 先取到分数，再清理
        rrf_score = item.get("_rrf_score", 0.0)
        rrf_key = item.pop("_doc_id", "")

        # 从 _key() 的编码格式反解析原始 doc_id（只有 "doc:" 前缀才有真实 doc_id）
        original_doc_id = rrf_key[4:] if rrf_key.startswith("doc:") else ""

        clean = {k2: v for k2, v in item.items() if k2 not in ("doc_id", "_rrf_score")}
        clean["doc_id"] = original_doc_id
        clean["_rrf_fuse_score"] = rrf_score
        clean["_fuse_source"] = "rrf"
        fused.append(clean)
    return fused


@track_node("rerank_node")
async def rerank_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """Rerank 重排序节点。

    返回：
        {"evidence_pool": [...], "rerank_stats": {...}, "messages": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("rerank", step="开始重排序", status="running")

    # 合并 web 和 local 证据
    web_evidence = state.get("web_evidence", [])
    local_evidence = state.get("local_evidence", [])

    logger.info(
        "[rerank] 待重排序证据总数=%d (web=%d, local=%d)",
        len(web_evidence) + len(local_evidence),
        len(web_evidence),
        len(local_evidence),
    )

    if not web_evidence and not local_evidence:
        logger.warning("[rerank] 无证据可重排序")
        if progress:
            progress("rerank", step="无证据可重排序", status="success")
        return {
            "rerank": "暂无可用证据需要重排序。",
            "evidence_pool": [],
            "rerank_stats": {"method": "none", "total_before": 0, "total_after": 0},
            "messages": [],
        }

    # 双路 RRF 融合：两路都出现的证据得分更高，排序更合理
    rrf_fused = _rrf_fuse_two_sources(web_evidence, local_evidence)
    logger.info("[rerank] RRF 融合后证据数=%d", len(rrf_fused))

    # 在传给 Reranker 之前清理内部融合字段，避免污染 Reranker 的输出
    _INTERNAL_KEYS = ("_rrf_fuse_score", "_fuse_source", "_doc_id", "_rrf_score")
    clean_fused = [
        {k: v for k, v in item.items() if k not in _INTERNAL_KEYS} for item in rrf_fused
    ]

    # 获取 query 用于相似度计算
    query = state.get("query", "")

    # 尝试导入 Reranker
    try:
        from ...retrieval.reranker import Reranker

        reranker = Reranker()
    except ImportError as e:
        logger.warning("[rerank] 导入 Reranker 失败: %s", e)
        reranker = None

    rerank_stats = {
        "method": "none",
        "total_before": len(web_evidence) + len(local_evidence),
        "total_after": 0,
        "rrf_fused_count": len(rrf_fused),
    }

    if reranker is not None:
        # 执行重排序（在 RRF 融合结果上进一步精排）
        try:
            reranked_results, stats = reranker.rerank(
                query=query,
                evidence_items=clean_fused,
                coarse_k=30,
                fine_k=10,
            )
            rerank_stats.update(stats)
            logger.info("[rerank] 重排序完成: %s", rerank_stats)
        except Exception as e:
            logger.warning("[rerank] 重排序失败，使用 RRF 融合结果: %s", e)
            reranked_results = clean_fused
            rerank_stats["method"] = "fallback"
    else:
        # 降级：直接用 RRF 融合结果
        reranked_results = clean_fused
        rerank_stats["method"] = (
            "rrf_fusion" if len(clean_fused) > 0 else "skipped_no_reranker"
        )

    rerank_stats["total_after"] = len(reranked_results)

    # 更新 source_type 标签（保留原始来源标记，清理内部融合字段）
    for idx, item in enumerate(reranked_results):
        if "source_type" not in item:
            item["source_type"] = "unknown"
        item.pop("_rrf_fuse_score", None)
        item.pop("_fuse_source", None)

    if progress:
        progress("rerank", step="重排序完成", status="success")

    return {
        "rerank": f"完成重排序，共 {len(reranked_results)} 条证据进入证据池。",
        "evidence_pool": reranked_results,
        "rerank_stats": rerank_stats,
        "messages": [],
    }
