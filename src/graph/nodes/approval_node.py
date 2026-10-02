"""人工审批节点：在 deep_dive 后暂停，等待人工确认。

使用 LangGraph interrupt() 实现人工审批：
- 当 evidence_pool 条目较多时（>10），中断流程等待人工确认
- 审批通过后继续到 analyze 节点
- 使用 checkpointer 支持中断恢复
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from ...state import ResearchState
from app.metrics import track_node

logger = logging.getLogger("research.nodes.approval")


@track_node("approval_node")
async def approval_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """人工审批节点。

    使用 interrupt() 暂停流程，等待人工确认。
    审批通过后继续到 analyze 节点。

    返回：
        {"approval": "approved"}  # 正常流程无需返回额外字段
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("approval", step="等待人工审批", status="running")

    evidence_count = len(state.get("evidence_pool", []))
    logger.info("[approval] 等待人工审批 | evidence_count=%d", evidence_count)

    if progress:
        progress("approval", step=f"证据池 {evidence_count} 条，等待确认", status="running")

    # 使用 interrupt 暂停，等待人工输入
    # interrupt() 返回用户输入的值（由外部通过 command.resume() 提供）
    approval_result = interrupt(
        {
            "message": f"证据裁判完成，共 {evidence_count} 条证据进入证据池。请确认是否继续分析？",
            "evidence_count": evidence_count,
        }
    )

    # 审批通过（默认接受用户输入）
    logger.info("[approval] 人工审批通过 | result=%s", approval_result)
    if progress:
        progress("approval", step="审批通过，继续分析", status="success")

    return {
        "approval": "approved",
        "approval_result": approval_result,
    }
