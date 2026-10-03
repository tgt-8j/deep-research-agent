"""写作节点：基于证据和结论生成最终研究报告。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.metrics import track_node

from ...prompt.loader import load_prompt
from ...state import ResearchState

logger = logging.getLogger("research.nodes.write")


def _extract_citation_ids(content: str) -> list[str]:
    """从正文中提取所有引用ID [XXX]。"""
    pattern = r"\[([A-Z]+\d+_\d+-\d+)\]"
    matches = re.findall(pattern, content)
    return list(dict.fromkeys(matches))  # 去重保序


def _validate_and_fix_citations(
    content: str, valid_source_ids: set[str]
) -> tuple[str, list[str]]:
    """校验正文中的引用ID，移除非法引用。"""
    pattern = r"\[([A-Z]+\d+_\d+-\d+)\]"

    def replace_citation(match):
        citation_id = match.group(1)
        if citation_id in valid_source_ids:
            return f"[{citation_id}]"
        return ""  # 移除非法引用

    fixed_content = re.sub(pattern, replace_citation, content)
    used_ids = [
        cid for cid in _extract_citation_ids(fixed_content) if cid in valid_source_ids
    ]
    return fixed_content, used_ids


def _render_reference_list(state: ResearchState) -> str:
    """渲染参考资料列表。"""
    lines = ["## 参考资料"]
    lookup = {}
    for source in state.get("source_index", []):
        sid = str(source.get("source_id", "")).strip()
        if sid:
            lookup[sid] = {
                "source_id": sid,
                "source_type": source.get("source_type", "source"),
                "label": source.get("label", sid),
                "locator": source.get("locator", ""),
            }

    cited_ids = []
    draft_content = state.get("draft", "") or state.get("final", "")
    if draft_content:
        for sid in _extract_citation_ids(draft_content):
            if sid in lookup and sid not in cited_ids:
                cited_ids.append(sid)

    if not cited_ids:
        for finding in state.get("findings", []):
            for sid in finding.get("source_ids", []):
                text = str(sid).strip()
                if text and text not in cited_ids and text in lookup:
                    cited_ids.append(text)

    if not cited_ids:
        cited_ids = list(lookup.keys())

    for sid in cited_ids[:15]:
        source = lookup.get(sid)
        if not source:
            continue
        locator = source.get("locator", "").strip() or (
            "链接暂不可用" if source["source_type"] == "web" else "本地知识库"
        )
        lines.append(
            f"- [{sid}] [{source['source_type']}]: {source['label']} | {locator}"
        )

    if len(lines) == 1:
        lines.append("- 暂无参考资料")
    return "\n".join(lines)


@track_node("write_node")
async def write_node(
    state: ResearchState,
    runtime: Any,
) -> dict[str, Any]:
    """写作节点：生成最终研究报告。

    返回：
        {"draft": str, "final": str}
    """
    progress = runtime.context.get("progress_emitter")
    if progress:
        progress("write", step="撰写研究报告", status="running")

    valid_source_ids = [
        str(item.get("source_id", "")).strip()
        for item in state.get("source_index", [])
        if item.get("source_id")
    ]
    valid_source_ids = [sid for sid in valid_source_ids if sid][:80]
    valid_source_ids_set = set(valid_source_ids)

    prompt_template = load_prompt("write")
    prompt = (
        f"{prompt_template}\n\n"
        f"核心问题：{state['query']}\n"
        f"子问题拆解：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n\n"
        f"【分析结论 (Findings)】：\n"
        f"{json.dumps(state.get('findings', []), ensure_ascii=False)}\n\n"
        f"【可用来源索引 (source_index)】：\n"
        f"{json.dumps(state.get('source_index', []), ensure_ascii=False)}\n\n"
        f"【合法引用ID列表】：\n"
        f"{json.dumps(valid_source_ids, ensure_ascii=False)}\n\n"
        f"【可能存在的风险/冲突 (Audit Flags)】：\n"
        f"{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}\n\n"
        f"要求：正文必须使用合法引用ID（例如 [WEB1_1-1]、[LOC1_1-3]）；禁止使用不存在的编号。"
        f"结尾不需要你来列举引用列表，系统会自动拼接。"
    )

    human = HumanMessage(content=prompt)
    llm = runtime.context.get("llm")
    messages = [human]

    if llm:
        result = await llm.ainvoke([human])
        content = result["messages"][-1].content
        messages = [human, result["messages"][-1]]
    else:
        content = f"# 研究报告\n\n## 摘要\n\n基于对「{state['query']}」的检索与分析，暂无可用证据支撑详细报告。\n\n请配置 LLM 后重试。"

    # 清理可能的 JSON 代码块
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```markdown\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"```$", "", content.strip())

    # 校验并修正引用
    content, _used_citation_ids = _validate_and_fix_citations(
        content, valid_source_ids_set
    )

    # 附加参考资料
    reference_list = _render_reference_list(state)
    final_content = f"{content.rstrip()}\n\n{reference_list}"

    if progress:
        progress("write", step="报告撰写完成", status="success")

    return {
        "draft": final_content,
        "final": final_content,
        "messages": messages,
    }
