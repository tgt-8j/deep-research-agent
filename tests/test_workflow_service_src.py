"""WorkflowService 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from src.config import AppConfig
from src.services.workflow import WorkflowService


def _make_config(**overrides) -> AppConfig:
    return AppConfig(
        api_key="test-key", model="qwen-plus", thread_id="t1",
        user_id="u1", tenant_id="tenant1", max_iterations=3,
        enable_memory=False, milvus_host="localhost", milvus_port=19530,
        milvus_collection="test", enable_milvus=False,
    )


class TestBuildState:
    def test_basic_fields(self):
        ws = WorkflowService(_make_config())
        state = ws._build_state("what is LangGraph", "user_42", "thread_7", "tenant_x")
        assert state["query"] == "what is LangGraph"
        assert state["user_id"] == "user_42"
        assert state["tenant_id"] == "tenant_x"
        assert state["phase"] == "initialized"
        assert state["cancellation"] is None

    def test_max_iterations_from_config(self):
        ws = WorkflowService(_make_config())
        state = ws._build_state("q", "u", "t", "tenant")
        assert state["max_iterations"] == 3

    def test_cancellation_injected(self):
        cancellation = MagicMock()
        ws = WorkflowService(_make_config())
        state = ws._build_state("q", "u", "t", "tenant", cancellation=cancellation)
        assert state["cancellation"] is cancellation

    def test_empty_strings_for_optional_fields(self):
        ws = WorkflowService(_make_config())
        state = ws._build_state("q", "u", "t", "tenant")
        assert state["intent"] == ""
        assert state["objective"] == ""
        assert state["draft"] == ""
        assert state["final"] == ""

    def test_thread_id_not_in_state(self):
        ws = WorkflowService(_make_config())
        state = ws._build_state("q", "u", "thread_7", "tenant")
        assert "thread_id" not in state


class TestEnsureInitialized:
    def test_graph_lazy_loaded(self):
        ws = WorkflowService(_make_config())
        assert ws._graph is None
        with patch("src.services.workflow.build_graph", return_value=MagicMock()) as mock_build:
            with patch("src.services.workflow.create_default_llm"):
                with patch("src.services.workflow.create_knowledge_base_client"):
                    ws._ensure_initialized()
        assert ws._graph is not None
        mock_build.assert_called_once()

    def test_idempotent(self):
        ws = WorkflowService(_make_config())
        with patch("src.services.workflow.build_graph", return_value=MagicMock()) as mock_build:
            with patch("src.services.workflow.create_default_llm"):
                with patch("src.services.workflow.create_knowledge_base_client"):
                    ws._ensure_initialized()
                    ws._ensure_initialized()
        assert mock_build.call_count == 1


class TestRun:
    @pytest.mark.asyncio
    async def test_run_returns_final_report(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"final": "# Report\n\nConclusion.", "intent": "multiagent"})
        ws._graph = mock_graph
        result = await ws.run("LangGraph agent", "u1", "t1", "tenant1")
        assert result == "# Report\n\nConclusion."
        mock_graph.ainvoke.assert_called_once()
        # config 以位置参数传入，call_args[0][1] 是第二个位置参数
        call_args = mock_graph.ainvoke.call_args[0]
        config = call_args[1] if len(call_args) > 1 else call_args.kwargs.get("config")
        assert config["configurable"]["thread_id"] == "t1"

    @pytest.mark.asyncio
    async def test_run_passes_cancellation(self):
        ws = WorkflowService(_make_config())
        cancellation = MagicMock()
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"final": "ok", "intent": "direct"})
        ws._graph = mock_graph
        await ws.run("q", "u1", "t1", "tenant1", cancellation=cancellation)
        state = mock_graph.ainvoke.call_args[0][0]
        assert state["cancellation"] is cancellation

    @pytest.mark.asyncio
    async def test_run_empty_final_defaults_to_empty_string(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={})
        ws._graph = mock_graph
        result = await ws.run("q", "u1", "t1", "tenant1")
        assert result == ""


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_success(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"final": "restored report", "intent": "multiagent"})
        ws._graph = mock_graph
        checkpoint = {"query": "old q", "phase": "rerank", "final": ""}
        result = await ws.resume("t1", checkpoint)
        assert result == "restored report"
        mock_graph.ainvoke.assert_called_once()
        restored_state = mock_graph.ainvoke.call_args[0][0]
        assert restored_state["query"] == "old q"
        assert restored_state["phase"] == "rerank"

    @pytest.mark.asyncio
    async def test_resume_fallback_on_error(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()
        call_count = [0]
        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("checkpoint not found")
            return {"final": "re-run result"}
        mock_graph.ainvoke = AsyncMock(side_effect=side_effect)
        ws._graph = mock_graph
        result = await ws.resume("t1", {"query": "q"})
        assert result == "re-run result"
        assert mock_graph.ainvoke.call_count == 2


class TestStreamEvents:
    @pytest.mark.asyncio
    async def test_stream_collects_events(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()

        async def fake_astream_events(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "intent"}
            yield {"event": "on_chain_end", "name": "intent", "data": {"output": {}}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "chunk1"}}}
            yield {"event": "on_chain_end", "name": "write", "data": {"output": {"final": "# Final", "intent": "multiagent"}}}

        mock_graph.astream_events = fake_astream_events
        ws._graph = mock_graph

        emitted = []
        await ws.stream_events("q", "u1", "t1", "tenant1", emit=emitted.append)

        types = [e["type"] for e in emitted]
        assert "status" in types
        assert "progress" in types
        assert "delta" in types
        assert "final" in types
        assert emitted[-1]["final"] == "# Final"

    @pytest.mark.asyncio
    async def test_stream_fallback_to_invoke_when_no_final(self):
        ws = WorkflowService(_make_config())
        mock_graph = MagicMock()

        async def fake_astream_empty(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "intent"}

        mock_graph.astream_events = fake_astream_empty
        mock_graph.ainvoke = AsyncMock(return_value={"final": "fallback report", "intent": "multiagent"})
        ws._graph = mock_graph

        emitted = []
        await ws.stream_events("q", "u1", "t1", "tenant1", emit=emitted.append)

        mock_graph.ainvoke.assert_called_once()
        final_event = next(e for e in emitted if e["type"] == "final")
        assert final_event["final"] == "fallback report"
