"""查询改写节点：将用户原始查询扩展为多个高召回率的搜索词。

在 plan_node 之前插入，解决用户提问口语化、关键词单一导致召回率低的问题。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...state import ResearchState

logger = logging.getLogger("research.nodes.query_rewrite")


def _default_rewrite(query: str) -> list[str]:
    """LLM 调用失败时的兜底改写策略。"""
    import re

    query.lower()
    # 提取核心实体
    chinese_terms = re.findall(r"[一-鿿]{2,}", query)
    english_terms = re.findall(r"[a-zA-Z]{3,}", query)
    entities = [
        t
        for t in chinese_terms + english_terms
        if t
        not in {
            "帮我",
            "调查",
            "最新",
            "使用趋势",
            "是什么",
            "多少",
            "情况",
            "关于",
            "这个",
            "那个",
            "如何",
            "怎么",
        }
    ]

    if not entities:
        return [query.strip()]

    entity = entities[0]
    variants = {query.strip()}
    # 中文变体
    if entities and any(ord(c) > 127 for c in entity):
        variants.add(f"{entity} 是什么")
        variants.add(f"{entity} 原理")
        variants.add(f"{entity} GitHub")
        variants.add(f"{entity} 使用教程")
    # 英文变体
    if english_terms:
        variants.add(f"{entity} tutorial")
        variants.add(f"{entity} documentation")
        variants.add(f"{entity} examples")
        variants.add(f"{entity} architecture")
    # 上位概念
    for word in ("framework", "library", "tool", "system", "platform"):
        variants.add(f"{entity} {word}")

    return list(variants)[:6]


@track_node("query_rewrite_node")
async def query_rewrite_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """查询改写节点。

    将用户原始查询扩展为多个搜索词变体，提升后续检索的召回率。

    返回：
        {"rewritten_queries": List[str], "messages": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("query_rewrite", step="改写搜索查询", status="running")

    query = state["query"]
    prompt_template = load_prompt("query_rewrite")
    prompt = f"{prompt_template}\n\n请改写以下查询：{query}"

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")
    messages = [human]

    if llm is None:
        logger.warning("[query_rewrite] LLM 未配置，使用兜底策略")
        rewritten = _default_rewrite(query)
    else:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                import re as _re

                cleaned = _re.sub(r"^```(?:json)?", "", cleaned).strip()
                cleaned = _re.sub(r"```$", "", cleaned).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(cleaned[start : end + 1])
                rewritten = parsed.get("rewritten_queries", [])
            else:
                rewritten = _default_rewrite(query)
        except (json.JSONDecodeError, Exception):
            rewritten = _default_rewrite(query)

    # 去重保序，限制数量
    seen = set()
    deduped = []
    for q in rewritten:
        q_stripped = q.strip()
        if q_stripped and q_stripped not in seen:
            seen.add(q_stripped)
            deduped.append(q_stripped)
    rewritten_queries = deduped[:6]

    logger.info(
        "[query_rewrite] 改写完成，生成 %d 个查询: %s",
        len(rewritten_queries),
        rewritten_queries,
    )

    if progress:
        progress(
            "query_rewrite",
            step=f"生成 {len(rewritten_queries)} 个查询变体",
            status="success",
        )

    return {
        "rewritten_queries": rewritten_queries,
        "messages": messages,
    }
