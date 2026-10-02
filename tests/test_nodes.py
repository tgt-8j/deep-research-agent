"""nodes.py 核心节点函数测试（纯函数）"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.nodes import (
    _extract_json_block,
    _load_json,
    _fallback_analysis,
    _render_fallback_report,
    _build_source_lookup,
    _assign_source_ids,
    _enrich_evidence_from_raw,
    _prune_evidence_to_allowed_sources,
    _dedupe_sources,
    _is_bad_web_domain,
    _filter_web_records,
    _format_raw_records,
    _minimal_record_filter,
    _normalize_source_ids,
    collect_tool_calls,
    _guess_primary_entity,
    _extract_query_terms,
    _estimate_relevance,
    _is_official_domain,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ── JSON 解析 ──────────────────────────────────────────────

class TestJsonHelpers:
    def test_extract_json_block_with_wrapping(self):
        text = "Here is the result:\n```json\n{\"key\": \"value\"}\n```"
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_extract_json_block_plain(self):
        text = '{"status": "ok", "count": 5}'
        result = _extract_json_block(text)
        assert result == '{"status": "ok", "count": 5}'

    def test_extract_json_block_no_json(self):
        text = "Just plain text"
        result = _extract_json_block(text)
        assert result == "Just plain text"

    def test_load_json_valid(self):
        text = '{"a": 1, "b": "two"}'
        result = _load_json(text, fallback={})
        assert result == {"a": 1, "b": "two"}

    def test_load_json_invalid_uses_fallback(self):
        result = _load_json("not json at all", {"default": True})
        assert result == {"default": True}


# ── 工具调用收集 ───────────────────────────────────────────

class TestCollectToolCalls:
    def test_no_tool_calls(self):
        messages = [HumanMessage(content="hello"), AIMessage(content="hi")]
        tools, outputs = collect_tool_calls(messages)
        assert tools == []
        assert outputs == []

    def test_with_tool_calls(self):
        messages = [
            HumanMessage(content="search"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "web_search",
                    "args": {"query": "test"},
                }],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
        tools, outputs = collect_tool_calls(messages)
        assert "web_search" in tools

    def test_multiple_tool_calls(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "tool_a", "args": {}},
                    {"id": "c2", "name": "tool_b", "args": {}},
                ],
            ),
        ]
        tools, outputs = collect_tool_calls(messages)
        assert "tool_a" in tools
        assert "tool_b" in tools


# ── 来源处理 ───────────────────────────────────────────────

class TestSourceProcessing:
    def test_assign_source_ids(self):
        records = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        result = _assign_source_ids(records, "WEB")
        assert result[0]["source_id"] == "WEB-1"
        assert result[1]["source_id"] == "WEB-2"
        assert result[2]["source_id"] == "WEB-3"

    def test_normalize_source_ids(self):
        values = ["WEB1_1", "LOC1_1", "WEB1_1", "WEB1_2"]
        result = _normalize_source_ids(values)
        assert len(result) == 3  # deduped

    def test_dedupe_sources(self):
        items = [
            {"source_id": "a", "title": "T1"},
            {"source_id": "a", "title": "T1"},
            {"source_id": "b", "title": "T2"},
        ]
        result = _dedupe_sources(items, ["source_id"])
        assert len(result) == 2

    def test_prune_evidence(self):
        evidence = [
            {"source_id": "WEB1_1", "claim": "a"},
            {"source_id": "WEB1_2", "claim": "b"},
        ]
        allowed = {"WEB1_1"}
        result = _prune_evidence_to_allowed_sources(evidence, allowed)
        assert len(result) == 1
        assert result[0]["source_id"] == "WEB1_1"

    def test_enrich_evidence(self):
        evidence = [{"source_id": "WEB1_1", "title": ""}]
        raw = [{"source_id": "WEB1_1", "title": "Actual Title"}]
        result = _enrich_evidence_from_raw(evidence, raw)
        assert result[0]["title"] == "Actual Title"


# ── 域名过滤 ───────────────────────────────────────────────

class TestDomainFilter:
    def test_bad_domain_with_blocked_keyword(self):
        assert _is_bad_web_domain("example-downsite.com") is True
        assert _is_bad_web_domain("doc88.cn") is True

    def test_good_domain(self):
        assert _is_bad_web_domain("github.com") is False
        assert _is_bad_web_domain("arxiv.org") is False

    def test_official_gov(self):
        assert _is_official_domain("www.example.gov.cn") is True
        assert _is_official_domain("www.university.edu.cn") is True
        assert _is_official_domain("github.com") is False


# ── 查询处理 ───────────────────────────────────────────────

class TestQueryProcessing:
    def test_extract_query_terms(self):
        terms = _extract_query_terms("LangGraph multi-agent system")
        assert isinstance(terms, list)
        assert len(terms) > 0

    def test_guess_primary_entity(self):
        entity = _guess_primary_entity("Tell me about LangGraph architecture")
        assert "langgraph" in entity.lower() or entity.lower() == "tell"

    def test_estimate_relevance_high(self):
        score = _estimate_relevance("Python programming", "Python is a programming language")
        assert score > 0

    def test_estimate_relevance_low(self):
        score = _estimate_relevance("quantum physics", "python cooking recipe")
        assert score < 0.3

    def test_filter_web_records(self):
        query = "machine learning"
        records = [
            {"snippet": "ML tutorial", "url": "https://example.com/ml"},
            {"snippet": "irrelevant spam", "url": "https://example.com/spam"},
        ]
        kept, rejected = _filter_web_records(query, records)
        assert isinstance(kept, list)
        assert isinstance(rejected, dict)


# ── 格式化 ─────────────────────────────────────────────────

class TestFormatting:
    def test_format_raw_records(self):
        records = [
            {"title": "Doc1", "snippet": "Content A", "source_type": "web"},
        ]
        result = _format_raw_records(records, "web")
        assert isinstance(result, str)
        assert "Doc1" in result

    def test_minimal_record_filter(self):
        records = [
            {"required_field": "yes", "title": "A"},
            {"required_field": "", "title": "B"},
        ]
        result = _minimal_record_filter(records, ["required_field"])
        assert len(result) == 1
        assert result[0]["title"] == "A"


# ── Fallback ───────────────────────────────────────────────

class TestFallbackFunctions:
    def test_fallback_analysis(self):
        state = {
            "query": "test question",
            "evidence_pool": [
                {"source_id": "WEB1_1", "title": "Source A"},
                {"source_id": "WEB1_2", "title": "Source B"},
            ],
            "hypotheses": [],
        }
        result = _fallback_analysis(state)
        assert "analysis_summary" in result
        assert "findings" in result

    def test_fallback_analysis_no_evidence(self):
        state = {"query": "test", "evidence_pool": [], "hypotheses": []}
        result = _fallback_analysis(state)
        assert "analysis_summary" in result

    def test_render_fallback_report(self):
        state = {
            "query": "test",
            "analysis": "Summary of findings",
            "hypotheses": [{"id": "h1", "content": "H1", "status": "verified"}],
            "findings": [{"claim": "Finding A", "source_ids": ["WEB1_1"]}],
        }
        report = _render_fallback_report(state)
        assert isinstance(report, str)
        assert "Summary of findings" in report

    def test_build_source_lookup(self):
        state = {
            "evidence_pool": [
                {"source_id": "WEB1_1", "title": "Title A", "url": "http://a.com", "source_type": "web"},
            ],
        }
        lookup = _build_source_lookup(state)
        assert isinstance(lookup, dict)
        assert "WEB1_1" in lookup
