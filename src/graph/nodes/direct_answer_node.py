"""直接回答节点：简单问题快速响应，不走调研流程。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from ...prompt.loader import load_prompt
from ...state import ResearchState
from app.metrics import track_node

logger = logging.getLogger("research.nodes.direct_answer")


@track_node("direct_answer_node")
async def direct_answer_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """直接回答节点。

    适用于问候、简单问答等不需要调研的问题。
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("direct_answer", step="直接回答问题", status="running")

    query = state["query"]
    prompt_template = load_prompt("direct_answer")
    prompt = f"{prompt_template}\n\n用户问题：{query}"

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")
    if llm is None:
        return {
            "intent": "direct",
            "final": "LLM 未配置，无法回答。",
            "messages": [human],
        }

    result = await llm.ainvoke([human])
    content = result["messages"][-1].content
    messages = [human, result["messages"][-1]]

    if progress:
        progress("direct_answer", step="回答完成", status="success")

    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": messages,
    }
