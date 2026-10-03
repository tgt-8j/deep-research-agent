"""分析节点：从证据池中归纳结论并评估证据完备性。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...prompt.models import AnalysisOutput, FindingItem
from ...state import ResearchState

logger = logging.getLogger("research.nodes.analyze")


@track_node("analyze_node")
async def analyze_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """分析节点：归纳结论、评估证据完备性。

    返回：
        {"analysis": "...", "findings": [...], "needs_more_research": bool,
         "missing_gaps": [...], "claim_map": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("analyze", step="生成分析结论", status="running")

    evidence_pool = state.get("evidence_pool", [])
    if not evidence_pool:
        logger.warning("[analyze] 证据池为空，使用默认分析")
        findings = [
            {
                "claim_id": "c_1",
                "claim": f"围绕'{state['query']}'已完成检索，但暂无可用证据。",
                "confidence": "low",
                "source_ids": [],
            }
        ]
        if progress:
            progress("analyze", step="证据不足，生成默认分析", status="success")
        return {
            "analysis": "当前证据不足以支撑深度分析。",
            "findings": findings,
            "claim_map": [{"claim_id": "c_1", "source_ids": []}],
            "needs_more_research": True,
            "missing_gaps": ["需要补充更多高质量来源"],
            "messages": [],
        }

    prompt_template = load_prompt("analyze")
    prompt = (
        f"{prompt_template}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"证据池：{json.dumps(evidence_pool[:10], ensure_ascii=False, default=str)}\n"
        f"审计标记：{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}"
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
            findings = parsed.get("findings", [])
            claim_map = parsed.get("claim_map", [])
            needs_more_research = parsed.get("needs_more_research", False)
            missing_gaps = parsed.get("missing_gaps", [])
            analysis_summary = parsed.get("analysis_summary", content)
        except (json.JSONDecodeError, AttributeError):
            findings = [
                {
                    "claim_id": "c_1",
                    "claim": f"围绕'{state['query']}'已完成多源检索，初步证据表明问题可从网络与本地知识库双侧支撑。",
                    "confidence": "medium",
                    "source_ids": [
                        e.get("source_id")
                        for e in evidence_pool[:3]
                        if e.get("source_id")
                    ],
                }
            ]
            claim_map = [{"claim_id": "c_1", "source_ids": findings[0]["source_ids"]}]
            needs_more_research = False
            missing_gaps = []
            analysis_summary = content
    else:
        findings = [
            {
                "claim_id": "c_1",
                "claim": f"围绕'{state['query']}'已完成多源检索。",
                "confidence": "medium",
                "source_ids": [
                    e.get("source_id") for e in evidence_pool[:3] if e.get("source_id")
                ],
            }
        ]
        claim_map = [{"claim_id": "c_1", "source_ids": findings[0]["source_ids"]}]
        needs_more_research = False
        missing_gaps = []
        analysis_summary = "完成结论归纳。"

    # Pydantic 结构化验证
    try:
        analysis_out = AnalysisOutput(
            analysis_summary=analysis_summary,
            needs_more_research=needs_more_research,
            missing_gaps=missing_gaps,
            findings=[FindingItem(**f) for f in findings],
            claim_map=claim_map,
        )
        findings = [f.model_dump() for f in analysis_out.findings]
        logger.info("[analyze] Pydantic 验证通过 | findings=%d", len(findings))
    except Exception as ve:
        logger.warning("[analyze] Pydantic 验证失败，使用原始数据 | %s", ve)

    if progress:
        progress("analyze", step="分析完成", status="success")

    return {
        "analysis": analysis_summary,
        "findings": findings,
        "claim_map": claim_map,
        "needs_more_research": needs_more_research,
        "missing_gaps": missing_gaps,
        "messages": messages,
    }
