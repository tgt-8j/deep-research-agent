"""证据裁判节点：对 Web 和本地证据进行评分、去重和冲突审计。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...prompt.models import AuditFlag, EvidencePool, EvidencePoolItem
from ...state import ResearchState

logger = logging.getLogger("research.nodes.deep_dive")


def _score_evidence(record: dict) -> tuple[float, str]:
    """根据来源类型和域名给证据打分。"""
    source_type = record.get("source_type")
    if source_type == "local":
        return 0.92, "企业内部知识库证据，默认高可信"
    domain = str(record.get("domain", "")).lower()
    if not domain:
        return 0.45, "来源信息不完整"
    if any(word in domain for word in [".gov.cn", ".gov", ".edu", ".edu.cn"]):
        return 0.88, "官方或权威机构域名"
    if any(
        word in domain
        for word in ["news", "finance", "reuters", "bloomberg", "people", "xinhuanet"]
    ):
        return 0.72, "主流媒体域名"
    return 0.58, "普通互联网来源，需要交叉验证"


def _fallback_evidence(records: list[dict], source_type: str) -> list[dict]:
    """将原始记录直接转为证据（LLM 失败时的兜底）。"""
    evidence = []
    for record in records:
        score, reason = _score_evidence(record)
        evidence.append(
            {
                "source_id": record.get("source_id", ""),
                "source_type": source_type,
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "doc_id": record.get("doc_id", ""),
                "snippet": record.get("snippet", "")[:500],
                "domain": record.get("domain", ""),
                "supports_questions": record.get("supports_questions", []),
                "reliability_score": score,
                "reliability_reason": reason,
                "source_label": record.get("title") or record.get("source_id", ""),
                "notes": "",
            }
        )
    return evidence


@track_node("deep_dive_node")
async def deep_dive_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """证据裁判节点。

    返回：
        {"evidence_pool": [...], "audit_flags": [...], "source_index": [...]}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("deep_dive", step="进行证据裁判", status="running")

    web_evidence = state.get("web_evidence", [])
    local_evidence = state.get("local_evidence", [])

    if not web_evidence and not local_evidence:
        logger.warning("[deep_dive] 等待检索结果：暂无可用证据")
        if progress:
            progress("deep_dive", step="无证据可裁判", status="success")
        return {
            "deep_dive": "暂无可用的证据需要进行裁判。",
            "evidence_pool": [],
            "audit_flags": [],
            "source_index": [],
        }

    # 合并所有证据
    all_records = web_evidence + local_evidence
    logger.info(
        "[deep_dive] 待裁判证据总数=%d (web=%d, local=%d)",
        len(all_records),
        len(web_evidence),
        len(local_evidence),
    )

    prompt_template = load_prompt("deep_dive")
    prompt = (
        f"{prompt_template}\n\n"
        f"问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"web_evidence：{json.dumps(web_evidence[:10], ensure_ascii=False, default=str)}\n"
        f"local_evidence：{json.dumps(local_evidence[:10], ensure_ascii=False, default=str)}"
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
            evidence_pool = parsed.get("evidence_pool", [])
            audit_flags = parsed.get("audit_flags", [])
        except (json.JSONDecodeError, AttributeError):
            evidence_pool = _fallback_evidence(all_records, "combined")
            audit_flags = []
    else:
        evidence_pool = _fallback_evidence(all_records, "combined")
        audit_flags = []

    # Pydantic 结构化验证
    try:
        evpool = EvidencePool(
            summary="证据裁判完成",
            evidence_pool=[EvidencePoolItem(**e) for e in evidence_pool],
            audit_flags=[AuditFlag(**f) for f in audit_flags],
        )
        evidence_pool = [e.model_dump() for e in evpool.evidence_pool]
        audit_flags = [f.model_dump() for f in evpool.audit_flags]
        logger.info(
            "[deep_dive] Pydantic 验证通过 | evidence=%d | flags=%d",
            len(evidence_pool),
            len(audit_flags),
        )
    except Exception as ve:
        logger.warning("[deep_dive] Pydantic 验证失败，使用原始数据 | %s", ve)

    # 构建 source_index
    source_index = []
    seen_ids = set()
    for ev in evidence_pool:
        sid = ev.get("source_id", "")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            source_index.append(
                {
                    "source_id": sid,
                    "label": ev.get("title") or ev.get("source_label", sid),
                    "locator": ev.get("url") or ev.get("doc_id", ""),
                    "source_type": ev.get("source_type", "source"),
                }
            )

    if progress:
        progress("deep_dive", step="证据裁判完成", status="success")

    return {
        "deep_dive": f"完成证据裁判，共 {len(evidence_pool)} 条证据进入证据池。",
        "audit": f"完成证据评分与审计，发现 {len(audit_flags)} 个审计标记。",
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": source_index,
        "messages": messages,
    }
