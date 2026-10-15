"""StreamAggregator 和 NodeReporter 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from src.presentation.aggregator import NodeReporter, StreamAggregator


class TestStreamAggregator:
    def test_initial_state(self):
        agg = StreamAggregator()
        snap = agg.snapshot()
        assert snap["messages"] == []
        assert snap["is_streaming"] is False
        assert snap["error"] is None

    def test_add_user_message(self):
        agg = StreamAggregator()
        msg = agg.add_user_message("hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert agg.is_streaming is True

    def test_delta_event_appends_to_assistant(self):
        agg = StreamAggregator()
        agg.apply({"type": "delta", "content": "hello "})
        agg.apply({"type": "delta", "content": "world"})
        assistant = [m for m in agg.messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].content == "hello world"

    def test_progress_event_does_not_add_message(self):
        agg = StreamAggregator()
        agg.apply({"type": "progress", "node": "intent", "status": "running"})
        assert len(agg.messages) == 0

    def test_reasoning_delta_appends_reasoning(self):
        agg = StreamAggregator()
        agg.apply({"type": "reasoning_delta", "content": "thinking..."})
        assert agg.messages[-1].reasoning == "thinking..."

    def test_reasoning_end_keeps_reasoning(self):
        agg = StreamAggregator()
        agg.apply({"type": "reasoning_delta", "content": "let me think"})
        agg.apply({"type": "reasoning_end"})
        assert agg.messages[-1].reasoning == "let me think"

    def test_message_event_adds_new_entry(self):
        agg = StreamAggregator()
        agg.apply({"type": "message", "role": "system", "content": "init"})
        assert agg.messages[-1].role == "system"
        assert agg.messages[-1].content == "init"

    def test_tool_call_event(self):
        agg = StreamAggregator()
        agg.apply(
            {"type": "tool_call", "tool": {"name": "search", "args": {"q": "test"}}}
        )
        assistant = agg.messages[-1]
        assert len(assistant.tool_calls) == 1
        assert assistant.tool_calls[0]["name"] == "search"

    def test_error_event(self):
        agg = StreamAggregator()
        agg.apply({"type": "error", "message": "connection lost"})
        assert agg.error is not None
        assert agg.is_streaming is False

    def test_final_event_stops_streaming(self):
        agg = StreamAggregator()
        agg.apply({"type": "delta", "content": "part1"})
        agg.apply({"type": "final", "content": "done"})
        assert agg.is_streaming is False
        assert agg.messages[-1].is_streaming is False

    def test_reset_clears_state(self):
        agg = StreamAggregator()
        agg.add_user_message("hi")
        agg.apply({"type": "delta", "content": "hello"})
        agg.reset()
        assert agg.messages == []
        assert agg.is_streaming is False
        assert agg.error is None
        assert agg._active_assistant_id is None

    def test_snapshot_format(self):
        agg = StreamAggregator()
        agg.add_user_message("q")
        agg.apply({"type": "delta", "content": "a"})
        snap = agg.snapshot()
        assert len(snap["messages"]) == 2
        assert snap["messages"][0]["role"] == "user"
        assert snap["messages"][1]["role"] == "assistant"
        assert snap["messages"][1]["content"] == "a"


class TestNodeReporter:
    def test_started_emits_progress(self):
        emitted = []
        reporter = NodeReporter("intent", "意图识别", emitted.append)
        reporter.started("开始分析意图")
        assert emitted[0]["type"] == "progress"
        assert emitted[0]["node"] == "intent"
        assert emitted[0]["status"] == "running"

    def test_completed_emits_success(self):
        emitted = []
        reporter = NodeReporter("write", "写作", emitted.append)
        reporter.completed("报告撰写完成")
        assert emitted[-1]["status"] == "success"

    def test_failed_emits_error_status(self):
        emitted = []
        reporter = NodeReporter("web_search", "网页搜索", emitted.append)
        reporter.failed("API timeout")
        assert emitted[-1]["status"] == "error"
        assert "timeout" in emitted[-1]["message"]

    def test_default_message_when_none(self):
        emitted = []
        reporter = NodeReporter("plan", "计划", emitted.append)
        reporter.started()
        assert len(emitted[0]["message"]) > 0
