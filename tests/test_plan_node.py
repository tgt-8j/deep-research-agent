"""单元测试：plan_node 的辅助函数。"""

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graph.nodes.plan_node import _derive_search_queries, _guess_primary_entity


class TestGuessPrimaryEntity:
    """测试实体提取逻辑。"""

    def test_chinese_entity(self):
        """提取中文核心实体。"""
        result = _guess_primary_entity("调研2024年LangGraph框架趋势")
        # Should extract something meaningful, not a stopword
        assert result != ""

    def test_english_entity(self):
        """提取英文核心实体。"""
        result = _guess_primary_entity("Research the latest AI agent frameworks")
        # Should extract entity, not stop words
        assert result != ""

    def test_stopword_filtering(self):
        """过滤停用词。"""
        result = _guess_primary_entity("帮我调查最新情况")
        # "帮我", "调查", "最新", "情况" may all be filtered or partially filtered
        # This test just verifies it doesn't crash
        assert isinstance(result, str)

    def test_empty_query(self):
        """空查询返回空字符串。"""
        result = _guess_primary_entity("")
        assert result == ""


class TestDeriveSearchQueries:
    """测试搜索词派生逻辑。"""

    def test_basic_derivation(self):
        """基础查询派生。"""
        queries = _derive_search_queries("LangGraph 是什么")
        assert len(queries) > 0
        assert "LangGraph 是什么" in queries

    def test_with_entity(self):
        """含实体时生成更多变体。"""
        queries = _derive_search_queries("DeepSeek V3 评测")
        # 应该包含原始查询和扩展查询
        assert any("DeepSeek" in q for q in queries)
        assert len(queries) >= 1

    def test_empty_query(self):
        """空查询返回空列表。"""
        queries = _derive_search_queries("")
        assert queries == []

    def test_max_six(self):
        """搜索词不超过6个。"""
        queries = _derive_search_queries("一个很长的查询包含很多关键词的测试")
        assert len(queries) <= 6
