"""网页搜索节点：执行网络检索并整理证据。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from ...context import ResearchRuntimeContext
from ...prompt.loader import load_prompt
from ...retrieval import bocha_web_search, _is_official_domain, _filter_records
from ...state import ResearchState
from app.metrics import track_node

logger = logging.getLogger("research.nodes.web_search")


def _assign_source_ids(records: list[dict], prefix: str) -> list[dict]:
    """为每条记录分配唯一的 source_id。"""
    assigned = []
    for idx, record in enumerate(records, 1):
        item = dict(record)
        item["source_id"] = f"{prefix}-{idx}"
        assigned.append(item)
    return assigned


def _build_queries(state: ResearchState, source_preference: str) -> list[dict]:
    """从 search_plan 或 supplementary_queries 中提取网页搜索查询。"""
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
        queries.append({
            "section_id": "sec_1",
            "query": state["query"],
            "source_preference": source_preference,
            "reason": "fallback",
        })
    return queries[:6]


@track_node("web_search_node")
async def web_search_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """网页搜索节点。

    返回：
        {"web_evidence": [...], "web_retrieval_stats": {...}, "web_search_trace": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("web_search", step="检索网络证据", status="running")

    queries = _build_queries(state, "web")
    iteration = state.get("iteration", 0)
    prefix = f"WEB{iteration + 1}"
    api_key = runtime.context.get("retrieval_config", {}).get("bocha_api_key")

    raw_records = []
    query_traces = list(state.get("web_search_trace", []))

    for query_idx, item in enumerate(queries, 1):
        query_text = str(item.get("query", ""))
        logger.info("[web_search] 执行查询 %d/%d: %s", query_idx, len(queries), query_text[:50])

        records = await bocha_web_search(query_text, count=4, api_key=api_key)
        records = _assign_source_ids(records, f"{prefix}_{query_idx}")
        for record in records:
            record["section_id"] = item.get("section_id", "sec_1")
            record["search_query"] = query_text

        raw_records.extend(records)
        query_traces.append({
            "iteration": iteration,
            "plan_step": query_idx,
            "query": query_text,
            "section_id": item.get("section_id", "sec_1"),
            "reason": item.get("reason", ""),
            "source_preference": "web",
            "raw_count": len(records),
            "raw_records": [{"source_id": r.get("source_id"), "title": r.get("title", "")[:50]}
                           for r in records[:3]],
        })

    # 去重
    seen = set()
    deduped = []
    for r in raw_records:
        key = (r.get("url") or r.get("source_id"), r.get("title"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    raw_records = deduped

    # 过滤低质量结果
    filtered_records, stats = _filter_records(state["query"], raw_records)

    web_retrieval_stats = dict(state.get("web_retrieval_stats", {}))
    web_retrieval_stats["query_count"] = web_retrieval_stats.get("query_count", 0) + len(queries)
    web_retrieval_stats["raw_count"] = web_retrieval_stats.get("raw_count", 0) + len(filtered_records)
    web_retrieval_stats["dropped_count"] = web_retrieval_stats.get("dropped_count", 0) + stats["dropped_irrelevant"]

    logger.info("[web_search] 去重过滤后记录数=%d", len(filtered_records))

    if not filtered_records:
        logger.warning("[web_search] 无可用网页证据")
        if progress:
            progress("web_search", step="无可用网页证据", status="success")
        return {
            "web_search": "未检索到可用网页证据。",
            "web_evidence": state.get("web_evidence", []),
            "web_retrieval_stats": web_retrieval_stats,
            "web_search_trace": query_traces,
        }

    # 调用 LLM 整理证据（简化版：直接结构化返回）
    prompt_template = load_prompt("web_search")
    prompt = (
        f"{prompt_template}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始网页证据：\n{json.dumps(filtered_records[:20], ensure_ascii=False, default=str)}"
    )

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")
    messages = [human]

    if llm:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
        # 解析 LLM 输出的证据
        try:
            parsed = json.loads(content)
            evidence = parsed.get("evidence", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: 将原始记录直接转为证据
            evidence = []
            for r in filtered_records:
                evidence.append({
                    "source_id": r.get("source_id", ""),
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", "")[:500],
                    "domain": r.get("domain", ""),
                    "source_type": "web",
                    "reliability_hint": "official" if _is_official_domain(r.get("domain", "")) else "unknown",
                    "supports_questions": [],
                    "notes": "",
                })
    else:
        evidence = []
        for r in filtered_records:
            evidence.append({
                "source_id": r.get("source_id", ""),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", "")[:500],
                "domain": r.get("domain", ""),
                "source_type": "web",
                "reliability_hint": "official" if _is_official_domain(r.get("domain", "")) else "unknown",
                "supports_questions": [],
                "notes": "",
            })

    # 限制证据数量
    evidence = evidence[:20]
    web_retrieval_stats["kept_count"] = web_retrieval_stats.get("kept_count", 0) + len(evidence)
    web_retrieval_stats["dropped_count"] = web_retrieval_stats.get("dropped_count", 0) + max(
        len(filtered_records) - len(evidence), 0
    )

    if progress:
        progress("web_search", step="网络证据整理完成", status="success")

    return {
        "web_search": f"完成网页证据采集，共 {len(evidence)} 条有效证据。",
        "web_evidence": state.get("web_evidence", []) + evidence,
        "web_retrieval_stats": web_retrieval_stats,
        "web_search_trace": query_traces,
        "messages": messages,
    }
