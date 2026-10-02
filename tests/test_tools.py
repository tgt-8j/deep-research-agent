"""tools.py 工具函数测试"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.tools import (
    _safe_path,
    _workspace_root,
    simple_calculator,
    summarize_points,
    dedupe_lines,
    merge_notes,
    extract_requirements,
    outline_from_topics,
    local_docs_lookup_stub,
)


def _call_tool(tool):
    """Helper to call a @tool-decorated function"""
    return tool.func if hasattr(tool, 'func') else tool


# ── 安全路径 ──────────────────────────────────────────────

class TestSafePath:
    def test_valid_relative_path(self, tmp_path):
        with patch.dict(os.environ, {"WORKSPACE_DIR": str(tmp_path)}):
            result = _safe_path("subdir/file.txt")
            assert result == tmp_path / "subdir" / "file.txt"

    def test_traversal_blocked(self, tmp_path):
        with patch.dict(os.environ, {"WORKSPACE_DIR": str(tmp_path)}):
            with pytest.raises(ValueError, match="路径超出工作目录"):
                _safe_path("../etc/passwd")

    def test_double_dot_blocked(self, tmp_path):
        with patch.dict(os.environ, {"WORKSPACE_DIR": str(tmp_path)}):
            with pytest.raises(ValueError, match="路径超出工作目录"):
                _safe_path("foo/../../bar")


# ── 文本处理工具 ───────────────────────────────────────────

class TestTextTools:
    def test_summarize_points(self):
        text = "点一：AI很重要。\n点二：Python是首选。"
        result = _call_tool(summarize_points)(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dedupe_lines(self):
        text = "line1\nline2\nline1\nline3\nline2"
        result = _call_tool(dedupe_lines)(text)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 3
        assert "line1" in lines

    def test_merge_notes(self):
        n1 = "要点A：\n- 内容1"
        n2 = "要点B：\n- 内容2"
        result = _call_tool(merge_notes)(n1, n2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_requirements(self):
        text = "项目需要：\n1. Python 3.10+\n2. LangGraph"
        result = _call_tool(extract_requirements)(text)
        assert isinstance(result, str)

    def test_outline_from_topics(self):
        text = "LangGraph, RAG, 多智能体"
        result = _call_tool(outline_from_topics)(text)
        assert isinstance(result, str)


# ── 工具 stub ─────────────────────────────────────────────

class TestStubTools:
    def test_local_docs_lookup_stub(self):
        result = _call_tool(local_docs_lookup_stub)("test query")
        assert "未配置" in result or "stub" in result.lower()
        assert "test query" in result


# ── 简单计算器 ─────────────────────────────────────────────

class TestCalculator:
    def test_basic_addition(self):
        result = _call_tool(simple_calculator)("2 + 3")
        assert "5" in str(result)

    def test_basic_subtraction(self):
        result = _call_tool(simple_calculator)("10 - 4")
        assert "6" in str(result)

    def test_basic_multiplication(self):
        result = _call_tool(simple_calculator)("3 * 7")
        assert "21" in str(result)

    def test_invalid_expression(self):
        with pytest.raises((ValueError, Exception)):
            _call_tool(simple_calculator)("abc")
