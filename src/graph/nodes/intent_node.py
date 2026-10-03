"""意图识别节点：判断用户问题是否需要深度调研。

参考 shopkeeper-agent 的 extract_keywords 模式：
- 先做规则初判，再用 LLM 微调决策
- 输出结构化 JSON 结果
- 通过 runtime.context.progress_emitter 上报进度
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...state import ResearchState

logger = logging.getLogger("research.nodes.intent")


def _rule_route(query: str) -> str:
    """基于关键词的规则初判，作为 LLM 决策的 fallback。

    包含以下关键词的问题默认走 multiagent 路径：
    - 调研类：调查、调研、来源、证据、检索统计
    - 分析类：趋势、新闻、盘点、榜单、对比、报告
    - 特定格式：包含年份+趋势/新闻等组合
    """
    force_multiagent_keywords = [
        "调查",
        "调研",
        "来源",
        "证据",
        "检索统计",
        "来源清单",
        "重大新闻",
        "热门项目",
        "趋势",
        "新闻",
        "最新",
        "盘点",
    ]
    normalized = query.strip()
    if re.search(r"20\d{2}年", normalized) and any(
        word in normalized for word in ["趋势", "新闻", "调研", "调查", "盘点"]
    ):
        return "multiagent"
    if any(word in query for word in force_multiagent_keywords):
        return "multiagent"

    keywords = [
        "调研",
        "研究",
        "调查",
        "盘点",
        "热门",
        "趋势",
        "榜单",
        "分析",
        "方案",
        "架构",
        "设计",
        "对比",
        "报告",
        "代码",
        "实现",
        "落地",
        "检索",
        "知识库",
        "证据",
        "来源",
        "溯源",
        "资料",
        "手册",
        "验证",
        "数据",
        "模型",
    ]
    return "multiagent" if any(word in query for word in keywords) else "direct"


def _with_memory_context(state: ResearchState, user_prompt: str) -> str:
    """将跨会话记忆注入到 prompt 中。"""
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"


@track_node("intent_node")
async def intent_node(
    state: ResearchState,
    runtime: Any,  # Runtime[ResearchRuntimeContext]
) -> dict[str, Any]:
    """意图识别节点。

    返回：
        {"intent": "direct" | "multiagent", "messages": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("intent", step="识别问题意图", status="running")

    query = state["query"]
    rule_route = _rule_route(query)
    logger.info("意图初判（规则）: %s | query=%s", rule_route, query[:50])

    prompt_template = load_prompt("intent_router")
    prompt = (
        f"{prompt_template}\n\n"
        f"用户问题：{query}\n"
        f"规则引擎初判：{rule_route}\n"
        f'请输出 JSON：{{"route":"direct|multiagent","reason":"..."}}'
    )

    human = HumanMessage(content=_with_memory_context(state, prompt))
    llm = runtime.context.get("llm")
    if llm is None:
        logger.warning("[intent] LLM 未配置，使用规则路由 fallback")
        route = rule_route
        messages = [human]
    else:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
        try:
            parsed = json.loads(content)
            route = parsed.get("route", rule_route)
        except (json.JSONDecodeError, AttributeError):
            route = rule_route

    if route not in {"direct", "multiagent"}:
        route = rule_route

    logger.info("意图路由: %s", route)
    if progress:
        progress("intent", step="意图识别完成", status="success")

    return {
        "intent": route,
        "messages": messages,
    }
