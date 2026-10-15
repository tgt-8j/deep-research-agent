"""rerank_node 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from src.graph.nodes.rerank_node import _rrf_fuse_two_sources, rerank_node


@pytest.fixture
def web_evidence():
    return [
        {
            "url": "https://a.com",
            "title": "A",
            "snippet": "s1",
            "source_id": "W1",
            "source_type": "web",
        },
        {
            "url": "https://b.com",
            "title": "B",
            "snippet": "s2",
            "source_id": "W2",
            "source_type": "web",
        },
    ]


@pytest.fixture
def local_evidence():
    return [
        {
            "doc_id": "d1",
            "title": "C",
            "snippet": "s3",
            "source_id": "L1",
            "source_type": "local",
        },
        {
            "doc_id": "d2",
            "title": "D",
            "snippet": "s4",
            "source_id": "L2",
            "source_type": "local",
        },
    ]


class TestRRFFuseTwoSources:
    def test_empty_inputs(self):
        assert _rrf_fuse_two_sources([], []) == []

    def test_web_only(self, web_evidence):
        result = _rrf_fuse_two_sources(web_evidence, [])
        assert len(result) == 2
        assert all(r.get("_fuse_source") == "single_source" for r in result)

    def test_local_only(self, local_evidence):
        result = _rrf_fuse_two_sources([], local_evidence)
        assert len(result) == 2

    def test_two_sources_merge(self, web_evidence, local_evidence):
        result = _rrf_fuse_two_sources(web_evidence, local_evidence)
        assert len(result) == 4
        source_ids = {r["source_id"] for r in result}
        assert source_ids == {"W1", "W2", "L1", "L2"}

    def test_shared_url_deduplicated(self):
        shared_url = "https://shared.com"
        web = [{"url": shared_url, "title": "S", "snippet": "s", "source_id": "W1"}]
        local = [
            {
                "doc_id": "d1",
                "url": shared_url,
                "title": "S",
                "snippet": "s",
                "source_id": "L1",
            }
        ]
        result = _rrf_fuse_two_sources(web, local)
        assert len(result) == 1
        assert result[0].get("_rrf_fuse_score", 0) > 0

    def test_single_source_zero_rrf_score(self, web_evidence):
        result = _rrf_fuse_two_sources(web_evidence, [])
        for item in result:
            assert item["_rrf_fuse_score"] == 0.0

    def test_rrf_fused_has_positive_score(self, web_evidence, local_evidence):
        result = _rrf_fuse_two_sources(web_evidence, local_evidence)
        rrf_items = [r for r in result if r.get("_fuse_source") == "rrf"]
        for item in rrf_items:
            assert item["_rrf_fuse_score"] > 0

    def test_source_id_preserved(self, web_evidence, local_evidence):
        result = _rrf_fuse_two_sources(web_evidence, local_evidence)
        assert all("source_id" in r for r in result)

    def test_source_id_key_path_covered(self):
        """items 无 url/doc_id 但有 source_id，触发 _key() 的 source_id 分支（lines 53-55）。"""
        web = [{"source_id": "W1", "title": "A", "snippet": "s"}]
        local = [{"source_id": "L1", "title": "B", "snippet": "s"}]
        result = _rrf_fuse_two_sources(web, local)
        assert len(result) == 2

    def test_title_key_path_covered(self):
        """items 无任何 key，触发 _key() 的 title fallback 分支（line 56）。"""
        web = [{"title": "A", "snippet": "s"}]
        local = [{"doc_id": "d1", "title": "B", "snippet": "s"}]
        result = _rrf_fuse_two_sources(web, local)
        assert len(result) == 2


def _make_runtime(progress_callback=None):
    runtime = MagicMock()
    runtime.context = {"progress_emitter": progress_callback}
    return runtime


def _make_state(web_ev=None, local_ev=None, query="LangGraph"):
    return {
        "query": query,
        "web_evidence": web_ev or [],
        "local_evidence": local_ev or [],
    }


class TestRerankNodeEmpty:
    @pytest.mark.asyncio
    async def test_no_evidence_returns_none_method(self):
        result = await rerank_node(_make_state(), _make_runtime())
        assert result["rerank_stats"]["method"] == "none"
        assert result["evidence_pool"] == []
        assert result["rerank_stats"]["total_before"] == 0

    @pytest.mark.asyncio
    async def test_progress_called_on_no_evidence(self):
        calls = []
        runtime = _make_runtime(progress_callback=lambda *a, **k: calls.append(k))
        await rerank_node(_make_state(), runtime)
        assert any("无证据" in c.get("step", "") for c in calls)


class TestRerankNodeWithEvidence:
    @pytest.mark.asyncio
    async def test_reranks_and_returns_pool(self):
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
            local_ev=[
                {
                    "doc_id": "d1",
                    "title": "B",
                    "snippet": "s",
                    "source_id": "L1",
                    "source_type": "local",
                }
            ],
        )
        result = await rerank_node(state, _make_runtime())
        assert len(result["evidence_pool"]) > 0
        assert result["rerank_stats"]["total_after"] > 0

    @pytest.mark.asyncio
    async def test_internal_keys_cleaned(self):
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
        )
        result = await rerank_node(state, _make_runtime())
        for item in result["evidence_pool"]:
            assert "_rrf_fuse_score" not in item
            assert "_fuse_source" not in item
            assert "_doc_id" not in item

    @pytest.mark.asyncio
    async def test_source_type_default_added(self):
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                }
            ],
        )
        result = await rerank_node(state, _make_runtime())
        for item in result["evidence_pool"]:
            assert "source_type" in item

    @pytest.mark.asyncio
    async def test_reranker_import_failure_fallback(self):
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
        )
        with patch(
            "src.retrieval.reranker.Reranker", side_effect=ImportError("no module")
        ):
            result = await rerank_node(state, _make_runtime())
        assert result["rerank_stats"]["method"] in ("rrf_fusion", "skipped_no_reranker")

    @pytest.mark.asyncio
    async def test_reranker_runtime_error_fallback(self):
        """Reranker.rerank() 抛异常时，降级到 RRF 融合结果（lines 177-180）。"""
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
        )
        with patch("src.retrieval.reranker.Reranker") as MockR:
            mock_instance = MagicMock()
            mock_instance.rerank.side_effect = RuntimeError("rerank failed")
            MockR.return_value = mock_instance
            result = await rerank_node(state, _make_runtime())
        assert result["rerank_stats"]["method"] == "fallback"

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        calls = []
        runtime = _make_runtime(progress_callback=lambda *a, **k: calls.append(k))
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
        )
        await rerank_node(state, runtime)
        assert any("重排序" in c.get("step", "") for c in calls)

    @pytest.mark.asyncio
    async def test_rerank_stats_structure(self):
        state = _make_state(
            web_ev=[
                {
                    "url": "https://a.com",
                    "title": "A",
                    "snippet": "s",
                    "source_id": "W1",
                    "source_type": "web",
                }
            ],
            local_ev=[
                {
                    "doc_id": "d1",
                    "title": "B",
                    "snippet": "s",
                    "source_id": "L1",
                    "source_type": "local",
                }
            ],
        )
        result = await rerank_node(state, _make_runtime())
        stats = result["rerank_stats"]
        assert all(
            k in stats
            for k in ("method", "total_before", "total_after", "rrf_fused_count")
        )
        assert stats["total_before"] == 2
