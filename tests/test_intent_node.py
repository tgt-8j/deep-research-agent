"""单元测试：intent_node 和 rule_route 逻辑。"""

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graph.nodes.intent_node import _rule_route


class TestRuleRoute:
    """测试规则引擎的意图路由判断。"""

    def test_simple_greeting(self):
        """简单问候应该路由到 direct。"""
        assert _rule_route("你好") == "direct"

    def test_who_are_you(self):
        """自我介绍查询应该路由到 direct。"""
        assert _rule_route("你是谁") == "direct"

    def test_weather_query(self):
        """简单问答应该路由到 direct。"""
        assert _rule_route("今天天气如何") == "direct"

    def test_research_keyword_trend(self):
        """包含趋势关键词应路由到 multiagent。"""
        assert _rule_route("调研2024年AI Agent市场趋势") == "multiagent"

    def test_research_keyword_source(self):
        """包含来源关键词应路由到 multiagent。"""
        assert _rule_route("帮我调查这个问题的证据来源") == "multiagent"

    def test_analysis_keyword(self):
        """包含分析关键词应路由到 multiagent。"""
        assert _rule_route("对比 LangGraph 和 CrewAI 的优缺点") == "multiagent"

    def test_report_keyword(self):
        """包含报告关键词应路由到 multiagent。"""
        assert _rule_route("写一篇关于 RAG 系统的深度报告") == "multiagent"

    def test_year_trend_combination(self):
        """年份+趋势组合应路由到 multiagent。"""
        assert _rule_route("2025年大模型应用趋势如何") == "multiagent"

    def test_deep_investigation(self):
        """深度调查应路由到 multiagent。"""
        assert _rule_route("深入调查 OpenAI 的最新动态") == "multiagent"

    def test_comparison(self):
        """对比查询应路由到 multiagent。"""
        assert _rule_route("RAG 和 Fine-tuning 哪种方案更适合我的场景") == "multiagent"

    def test_empty_query(self):
        """空查询默认路由到 direct。"""
        assert _rule_route("") == "direct"
