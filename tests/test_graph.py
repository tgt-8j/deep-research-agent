"""graph.py 工作流编排测试"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.graph import route_after_intent, should_continue_research, build_app


class TestRouteAfterIntent:
    def test_direct_route(self):
        state = {"intent": "direct", "query": "简单问题"}
        result = route_after_intent(state)
        assert result == "direct_answer"

    def test_multi_agent_route(self):
        state = {"intent": "multiagent", "query": "复杂研究"}
        result = route_after_intent(state)
        assert result == "plan"

    def test_unknown_intent_defaults_to_plan(self):
        state = {"intent": "unknown", "query": "test"}
        result = route_after_intent(state)
        assert result == "plan"


class TestShouldContinueResearch:
    def test_reached_max_iterations(self):
        state = {"iteration": 3, "max_iterations": 2, "needs_more_research": False}
        result = should_continue_research(state)
        assert result == "write"

    def test_needs_more_research(self):
        state = {"iteration": 1, "max_iterations": 3, "needs_more_research": True}
        result = should_continue_research(state)
        assert result == "reflect"

    def test_enough_evidence(self):
        state = {"iteration": 1, "max_iterations": 3, "needs_more_research": False}
        result = should_continue_research(state)
        assert result == "write"

    def test_default_state(self):
        state = {}
        result = should_continue_research(state)
        assert result == "write"


class TestBuildApp:
    def test_builds_graph(self):
        mock_agents = MagicMock()
        mock_agents.intent_router = MagicMock()
        mock_agents.direct_responder = MagicMock()
        mock_agents.planner = MagicMock()
        mock_agents.scout_web = MagicMock()
        mock_agents.scout_local = MagicMock()
        mock_agents.evidence_judge = MagicMock()
        mock_agents.analyst = MagicMock()
        mock_agents.writer = MagicMock()
        # Use InMemorySaver instead of MagicMock for checkpointer
        from langgraph.checkpoint.memory import InMemorySaver
        app = build_app(mock_agents, InMemorySaver())
        assert app is not None
