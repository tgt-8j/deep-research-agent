"""ResearchRequest 校验测试"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend.schemas.research import ResearchRequest


class TestResearchRequestValidation:
    """请求体校验测试。"""

    def test_valid_request(self):
        req = ResearchRequest(query="测试问题")
        assert req.query == "测试问题"
        assert req.user_id == "default_user"
        assert req.tenant_id == "default_tenant"

    def test_custom_fields(self):
        req = ResearchRequest(
            query="复杂研究问题",
            user_id="user123",
            tenant_id="tenant_a",
            max_iterations=5,
        )
        assert req.query == "复杂研究问题"
        assert req.max_iterations == 5

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            ResearchRequest(query="")

    def test_query_max_length(self):
        long_query = "x" * 4001
        with pytest.raises(Exception):
            ResearchRequest(query=long_query)

    def test_query_normal_length_accepted(self):
        normal_query = "x" * 4000
        req = ResearchRequest(query=normal_query)
        assert len(req.query) == 4000

    @pytest.mark.parametrize("value", ["user with spaces", "user@domain", "user/name"])
    def test_invalid_user_id_format(self, value):
        with pytest.raises(Exception):
            ResearchRequest(user_id=value)

    def test_valid_user_id(self):
        req = ResearchRequest(query="test", user_id="user_123-test")
        assert req.user_id == "user_123-test"

    def test_user_id_max_length(self):
        long_id = "x" * 65
        with pytest.raises(Exception):
            ResearchRequest(user_id=long_id)

    def test_max_iterations_bounds(self):
        with pytest.raises(Exception):
            ResearchRequest(max_iterations=0)
        with pytest.raises(Exception):
            ResearchRequest(max_iterations=7)

    def test_max_iterations_valid(self):
        req = ResearchRequest(query="test", max_iterations=1)
        assert req.max_iterations == 1
        req = ResearchRequest(query="test", max_iterations=6)
        assert req.max_iterations == 6


class TestResearchResponse:
    def test_response_model(self):
        from backend.schemas.research import ResearchResponse

        resp = ResearchResponse(
            query="test",
            user_id="u1",
            thread_id="t1",
            tenant_id="ten1",
            final="最终答案",
        )
        assert resp.final == "最终答案"
        assert resp.query == "test"
