"""本地知识库检索节点。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...retrieval import KnowledgeBaseClient
from ...state import ResearchState

logger = logging.getLogger("research.nodes.local_rag")


def _build_queries(state: ResearchState, source_preference: str) -> list[dict]:
    """从 search_plan 或 supplementary_queries 中提取本地检索查询。"""
    queries: list[dict] = []
    iteration = state.get("iteration", 0)
    if iteration > 0 and state.get("supplementary_queries"):
        base_plan = state.get("supplementary_queries", [])
    else:
        base_plan = state.get("search_plan", [])

    for item in base_plan:
        if not isinstance(item, dict):
            continue
        pref = item.get("source_preference", "hybrid")
        if pref in (source_preference, "hybrid"):
            query_text = str(item.get("query", "")).strip()
            if query_text:
                queries.append(item)
    if not queries:
        queries.append(
            {
                "section_id": "sec_1",
                "query": state["query"],
                "source_preference": source_preference,
                "reason": "fallback",
            }
        )
    return queries[:6]


@track_node("local_rag_node")
async def local_rag_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """本地知识库检索节点。

    返回：
        {"local_evidence": [...], "local_retrieval_stats": {...}, "local_rag_trace": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("local_rag", step="检索本地知识库", status="running")

    queries = _build_queries(state, "local")
    iteration = state.get("iteration", 0)
    prefix = f"LOC{iteration + 1}"

    kb_client: KnowledgeBaseClient = runtime.context.get("kb_client")
    if kb_client is None:
        logger.warning("[local_rag] 知识库客户端未配置，跳过本地检索")
        if progress:
            progress("local_rag", step="本地检索不可用", status="error")
        return {
            "local_rag": "本地知识库未配置，已跳过本地上下文注入。",
            "local_evidence": state.get("local_evidence", []),
            "local_retrieval_stats": {},
            "local_rag_trace": [],
        }

    raw_records = []
    query_traces = list(state.get("local_rag_trace", []))
    stats_extra = {"hybrid_count": 0, "bm25_count": 0}

    for query_idx, item in enumerate(queries, 1):
        query_text = str(item.get("query", ""))
        logger.info(
            "[local_rag] 执行查询 %d/%d: %s", query_idx, len(queries), query_text[:50]
        )

        # 先尝试混合检索（BM25 + 向量 RRF 融合）
        hybrid_records = []
        if hasattr(kb_client, "hybrid_search"):
            try:
                hybrid_records = kb_client.hybrid_search(query_text, limit=4)
                stats_extra["hybrid_count"] += len(hybrid_records)
            except Exception as e:
                logger.warning("[local_rag] 混合检索失败，降级为普通检索: %s", e)

        # 混合检索无结果时降级到普通检索
        if not hybrid_records:
            records = kb_client.search(query_text, limit=4)
            stats_extra["bm25_count"] += 0  # 标记使用了降级
        else:
            records = hybrid_records
            stats_extra["bm25_count"] += len(hybrid_records)

        for record in records:
            record["source_id"] = f"{prefix}_{query_idx}-{len(raw_records) + 1}"
            record["section_id"] = item.get("section_id", "sec_1")
            record["search_query"] = query_text
        raw_records.extend(records)

        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_idx,
                "query": query_text,
                "section_id": item.get("section_id", "sec_1"),
                "reason": item.get("reason", ""),
                "source_preference": "local",
                "raw_count": len(records),
                "raw_records": [
                    {"source_id": r.get("source_id"), "title": r.get("title", "")[:50]}
                    for r in records[:3]
                ],
            }
        )

    # 去重
    seen = set()
    deduped = []
    for r in raw_records:
        key = (r.get("doc_id") or r.get("source_id"), r.get("snippet", "")[:100])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    raw_records = deduped

    local_retrieval_stats = dict(state.get("local_retrieval_stats", {}))
    local_retrieval_stats["query_count"] = local_retrieval_stats.get(
        "query_count", 0
    ) + len(queries)
    local_retrieval_stats["raw_count"] = local_retrieval_stats.get(
        "raw_count", 0
    ) + len(raw_records)
    local_retrieval_stats["hybrid_count"] = stats_extra.get("hybrid_count", 0)
    local_retrieval_stats["bm25_count"] = stats_extra.get("bm25_count", 0)

    logger.info("[local_rag] 去重后记录数=%d", len(raw_records))

    if not raw_records:
        logger.warning("[local_rag] 无可用本地证据")
        if progress:
            progress("local_rag", step="无可用本地证据", status="success")
        return {
            "local_rag": "未检索到可用本地知识库证据，已跳过本地上下文注入。",
            "local_evidence": state.get("local_evidence", []),
            "local_retrieval_stats": local_retrieval_stats,
            "local_rag_trace": query_traces,
        }

    # 调用 LLM 整理本地证据
    prompt_template = load_prompt("local_rag")
    prompt = (
        f"{prompt_template}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始知识库证据：\n{json.dumps(raw_records[:20], ensure_ascii=False, default=str)}"
    )

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")
    messages = [human]

    if llm:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
        try:
            parsed = json.loads(content)
            evidence = parsed.get("evidence", [])
        except (json.JSONDecodeError, AttributeError):
            evidence = []
            for r in raw_records:
                evidence.append(
                    {
                        "source_id": r.get("source_id", ""),
                        "doc_id": r.get("doc_id", ""),
                        "title": r.get("title") or r.get("source_id", ""),
                        "snippet": r.get("snippet", "")[:500],
                        "source_type": "local",
                        "reliability_hint": "internal",
                        "supports_questions": [],
                        "notes": "",
                    }
                )
    else:
        evidence = []
        for r in raw_records:
            evidence.append(
                {
                    "source_id": r.get("source_id", ""),
                    "doc_id": r.get("doc_id", ""),
                    "title": r.get("title") or r.get("source_id", ""),
                    "snippet": r.get("snippet", "")[:500],
                    "source_type": "local",
                    "reliability_hint": "internal",
                    "supports_questions": [],
                    "notes": "",
                }
            )

    evidence = evidence[:20]
    local_retrieval_stats["kept_count"] = local_retrieval_stats.get(
        "kept_count", 0
    ) + len(evidence)
    local_retrieval_stats["dropped_count"] = local_retrieval_stats.get(
        "dropped_count", 0
    ) + max(len(raw_records) - len(evidence), 0)

    if progress:
        progress("local_rag", step="本地证据整理完成", status="success")

    return {
        "local_rag": f"完成本地知识库证据采集，共 {len(evidence)} 条有效证据。",
        "local_evidence": state.get("local_evidence", []) + evidence,
        "local_retrieval_stats": local_retrieval_stats,
        "local_rag_trace": query_traces,
        "messages": messages,
    }
