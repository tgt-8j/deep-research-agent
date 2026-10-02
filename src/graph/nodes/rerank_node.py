"""Rerank 重排序节点：对 web/local 两路检索结果进行粗筛+精排。

职责：
- 合并 web_evidence + local_evidence
- 用 Embedding 向量相似度做两级重排序（粗筛 Top-30，精排 Top-10）
- 输出 rerank_stats 统计信息和重新排序后的 evidence_pool
- 降级策略：Embedding 不可用时跳过重排序，直接返回原始合并结果

设计原则：
- 不修改原始证据数据，只改变排序
- rerank_stats 记录重排序过程，便于调试和面试讲解
- 与 deep_dive_node 分离：rerank 只做排序，deep_dive 做评分/去重/冲突检测
"""

from __future__ import annotations

import logging
from typing import Any

from app.metrics import track_node

from ...state import ResearchState

logger = logging.getLogger("research.nodes.rerank")


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
    all_evidence = web_evidence + local_evidence

    logger.info(
        "[rerank] 待重排序证据总数=%d (web=%d, local=%d)",
        len(all_evidence),
        len(web_evidence),
        len(local_evidence),
    )

    if not all_evidence:
        logger.warning("[rerank] 无证据可重排序")
        if progress:
            progress("rerank", step="无证据可重排序", status="success")
        return {
            "rerank": "暂无可用证据需要重排序。",
            "evidence_pool": [],
            "rerank_stats": {"method": "none", "total_before": 0, "total_after": 0},
            "messages": [],
        }

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
        "total_before": len(all_evidence),
        "total_after": 0,
    }

    if reranker is not None:
        # 执行重排序
        try:
            reranked_results, stats = reranker.rerank(
                query=query,
                evidence_items=all_evidence,
                coarse_k=30,
                fine_k=10,
            )
            rerank_stats.update(stats)
            logger.info("[rerank] 重排序完成: %s", rerank_stats)
        except Exception as e:
            logger.warning("[rerank] 重排序失败，使用原始顺序: %s", e)
            reranked_results = all_evidence
            rerank_stats["method"] = "fallback"
    else:
        # 降级：直接返回原始合并结果
        reranked_results = all_evidence
        rerank_stats["method"] = "skipped_no_reranker"

    rerank_stats["total_after"] = len(reranked_results)

    # 更新 source_type 标签（保留原始来源标记）
    for idx, item in enumerate(reranked_results):
        if "source_type" not in item:
            item["source_type"] = "unknown"

    if progress:
        progress("rerank", step="重排序完成", status="success")

    return {
        "rerank": f"完成重排序，共 {len(reranked_results)} 条证据进入证据池。",
        "evidence_pool": reranked_results,
        "rerank_stats": rerank_stats,
        "messages": [],
    }
