"""规划节点：将用户问题拆解为子问题和搜索计划。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...state import ResearchState

logger = logging.getLogger("research.nodes.plan")


def _default_plan(state: ResearchState) -> dict:
    """规划节点的兜底结果，当 LLM 调用失败时使用。"""
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {
            "max_rounds": 2,
            "max_sources": 12,
            "max_tokens": 12000,
            "max_seconds": 45,
        },
    }


def _guess_primary_entity(query: str) -> str:
    """从查询中提取核心实体，用于生成搜索词。"""
    lowered = query.lower()
    # 尝试匹配英文实体
    import re

    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "agent", "open", "using"}:
            return term
    # 尝试匹配中文实体
    chinese_terms = re.findall(r"[一-鿿]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""


def _derive_search_queries(query: str) -> list[str]:
    """基于原始查询派生出多个搜索词。"""
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    candidates = [base_query]
    if entity:
        candidates.extend(
            [
                f"{entity}是什么",
                f"{entity} GitHub",
                f"{entity} 官方文档",
                f"{entity} 使用趋势",
                f"{entity} AI Agent",
            ]
        )
    else:
        candidates.extend(
            [
                f"{base_query} 是什么",
                f"{base_query} GitHub",
                f"{base_query} 官方文档",
            ]
        )
    # 去重保序
    deduped = []
    for item in candidates:
        text = item.strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:6]


@track_node("plan_node")
async def plan_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """规划节点：拆解问题、生成大纲和搜索计划。

    返回：
        {"outline": [...], "sub_questions": [...], "search_plan": [...], ...}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("plan", step="拆解研究问题", status="running")

    query = state["query"]
    prompt_template = load_prompt("plan")
    prompt = (
        f"{prompt_template}\n\n"
        f"用户需求：{query}\n"
        f"请先做大纲与问题拆解，再输出规划 JSON。"
    )

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")

    if llm is None:
        logger.warning("[plan] LLM 未配置，使用默认规划")
        fallback = _default_plan(state)
        messages = [human]
    else:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
        try:
            # 尝试从内容中提取 JSON
            cleaned = content.strip()
            if cleaned.startswith("```"):
                import re

                cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
                cleaned = re.sub(r"```$", "", cleaned).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                fallback = json.loads(cleaned[start : end + 1])
            else:
                fallback = _default_plan(state)
        except (json.JSONDecodeError, Exception):
            fallback = _default_plan(state)

    # 从 LLM 输出或 fallback 中提取结构化字段
    outline = fallback.get("outline", _default_plan(state)["outline"])
    sub_questions = fallback.get("sub_questions", _default_plan(state)["sub_questions"])
    research_questions = fallback.get(
        "research_questions", _default_plan(state)["research_questions"]
    )
    budget = fallback.get("budget", _default_plan(state)["budget"])

    # 生成搜索计划
    search_plan = []
    for query_text in _derive_search_queries(query):
        search_plan.append(
            {
                "section_id": "user_query",
                "query": query_text,
                "source_preference": "hybrid",
                "reason": "围绕用户原始问题生成的直接检索词",
            }
        )
    # 从大纲章节中提取搜索词
    for section in outline:
        if not isinstance(section, dict):
            continue
        for sq in section.get("search_queries", []):
            search_plan.append(
                {
                    "section_id": section.get("id", "sec"),
                    "query": sq,
                    "source_preference": "hybrid",
                    "reason": f"来自大纲章节 {section.get('id', 'sec')}",
                }
            )

    objective = fallback.get("objective", query)

    if progress:
        progress("plan", step="规划完成", status="success")

    return {
        "phase": "planning completed",
        "plan": objective,
        "outline": outline,
        "sub_questions": sub_questions,
        "research_questions": research_questions,
        "search_plan": search_plan,
        "budget": budget,
        "messages": messages,
        "draft": str(fallback),
        "iteration": 0,
    }
