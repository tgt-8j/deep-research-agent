"""反思节点：基于信息缺口生成补搜计划。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...state import ResearchState

logger = logging.getLogger("research.nodes.reflect")


@track_node("reflect_node")
async def reflect_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """反思节点：生成补搜计划以填补缺口。

    返回：
        {"iteration": int, "supplementary_queries": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("reflect", step="分析信息缺口", status="running")

    missing_gaps = state.get("missing_gaps", [])
    if not missing_gaps:
        logger.info("[reflect] 无信息缺口，直接跳至 analyze 节点")
        if progress:
            progress("reflect", step="无需补搜，跳至分析", status="success")
        # 使用 Command 跨节点跳转，跳过 web_search/local_rag
        return Command(goto="analyze")

    logger.info("[reflect] 信息缺口=%s", missing_gaps)

    prompt_template = load_prompt("reflect")
    prompt = (
        f"{prompt_template}\n\n"
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps, ensure_ascii=False)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []), ensure_ascii=False)}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []), ensure_ascii=False)}\n\n"
        f"请生成新的、针对性的搜索词以填补这些缺口。"
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
            supplementary_queries = parsed.get("supplementary_queries", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback：基于缺失缺口生成简单查询
            supplementary_queries = [
                {
                    "section_id": f"gap_{i + 1}",
                    "query": gap,
                    "source_preference": "hybrid",
                    "reason": f"填补缺口: {gap}",
                }
                for i, gap in enumerate(missing_gaps)
            ]
    else:
        supplementary_queries = [
            {
                "section_id": f"gap_{i + 1}",
                "query": gap,
                "source_preference": "hybrid",
                "reason": f"填补缺口: {gap}",
            }
            for i, gap in enumerate(missing_gaps)
        ]

    new_iteration = state.get("iteration", 0) + 1

    if progress:
        progress(
            "reflect",
            step=f"补搜计划生成完成（第 {new_iteration} 轮）",
            status="success",
        )

    return {
        "iteration": new_iteration,
        "supplementary_queries": supplementary_queries,
        "messages": messages,
    }
