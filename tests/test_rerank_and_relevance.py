"""两级 rerank 与检索相关性的回归测试。

锁住两个已经修掉的缺陷：

1. **rerank 的两级原来是同一个信号**：粗筛和精排都算 ``title+snippet`` 的
   embedding 余弦，精排只会得到和粗筛一样的顺序，还多付一次 embedding 费用。
   现在粗筛是 BM25 词法（不碰 embedding），精排才是语义 —— 本文件用
   "词法命中" 与 "语义相似" 故意错位的样本证明两级会给出不同顺序。

2. **中文相关性过滤与域名判定**：
   - 旧 tokenizer ``[一-鿿]{2,}`` 把整个中文问句当成一个 term，相关结果成批被丢；
   - 旧 ``_is_official_domain`` 用子串 ``"gov" in domain``，``government-scam.com``
     会被当成权威来源。
"""

from __future__ import annotations

from src.retrieval.reranker import Reranker
from src.retrieval.web_search import (
    _estimate_relevance,
    _extract_query_terms,
    _filter_records,
    _is_official_domain,
)

LONG_CJK_QUERY = "2025年中国新能源汽车出口量是多少"


def _item(source_id: str, title: str, snippet: str = "", **extra) -> dict:
    return {"source_id": source_id, "title": title, "snippet": snippet, **extra}


# ---------------------------------------------------------------------------
# 1. 中文分词 / 相关性
# ---------------------------------------------------------------------------


class TestChineseQueryTerms:
    def test_long_query_is_split_into_bigrams(self):
        terms = _extract_query_terms(LONG_CJK_QUERY)
        # 不能是"整句一个 term"
        assert LONG_CJK_QUERY not in terms
        # 关键 bigram 要出现
        assert "新能" in terms
        assert "出口" in terms
        assert "2025" in terms
        assert len(terms) >= 6

    def test_short_chinese_query(self):
        assert "芯片" in _extract_query_terms("芯片")

    def test_english_terms_kept_whole(self):
        terms = _extract_query_terms("LangGraph reflection loop")
        assert "langgraph" in terms
        assert "reflection" in terms

    def test_relevant_chinese_snippet_scores_high(self):
        """相关的中文片段必须拿到足够高的分数（旧实现恒为 0）。"""
        score = _estimate_relevance(
            LONG_CJK_QUERY, "2025年我国新能源汽车出口量持续增长，达到200万辆"
        )
        assert score >= 0.5

    def test_unrelated_snippet_scores_low(self):
        assert _estimate_relevance(LONG_CJK_QUERY, "Python 是一门编程语言") < 0.2


class TestChineseFiltering:
    def test_relevant_chinese_record_kept(self):
        records = [
            {
                "title": "新能源汽车出口保持增长",
                "snippet": "2025年我国新能源汽车出口量持续增长",
                "domain": "auto.example.com",
            },
            {
                "title": "Python 基础教程",
                "snippet": "列表推导式与生成器",
                "domain": "py.example.com",
            },
        ]
        kept, stats = _filter_records(LONG_CJK_QUERY, records)
        assert len(kept) == 1
        assert kept[0]["title"] == "新能源汽车出口保持增长"
        assert stats["raw_count"] == 2
        assert stats["dropped_irrelevant"] == 1
        # 阈值可被调用方读取，便于排查"为什么被丢了"
        assert stats["min_relevance"] == 0.2


# ---------------------------------------------------------------------------
# 2. 官方域名判定
# ---------------------------------------------------------------------------


class TestOfficialDomain:
    def test_common_official(self):
        for domain in ("www.gov.cn", "example.gov.cn", "pku.edu.cn", "MIT.EDU"):
            assert _is_official_domain(domain) is True, domain

    def test_substring_trap_is_gone(self):
        """旧实现里这三个都会被判成权威来源。"""
        for domain in (
            "government-scam.com",
            "unofficial-news.com",
            "mygov.example.com",
        ):
            assert _is_official_domain(domain) is False, domain

    def test_normal_and_media(self):
        assert _is_official_domain("github.com") is False
        assert _is_official_domain("news.baidu.com") is False

    def test_tolerates_url_and_port(self):
        assert _is_official_domain("https://www.example.gov.cn/news/1") is True
        assert _is_official_domain("example.edu.cn:8080") is True

    def test_empty(self):
        assert _is_official_domain("") is False

    def test_evidence_scoring_uses_suffix_rule(self):
        """证据裁判也应复用同一套后缀判定。"""
        from src.graph.nodes.deep_dive_node import _score_evidence

        assert _score_evidence({"source_type": "web", "domain": "a.gov.cn"})[0] == 0.88
        assert (
            _score_evidence({"source_type": "web", "domain": "government-scam.com"})[0]
            == 0.58
        )


# ---------------------------------------------------------------------------
# 3. 两级 rerank 用的是不同信号
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    """按文本返回固定向量，避免测试打真实 embedding 接口。"""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.mapping.get(text, [0.0, 0.0]) for text in texts]


class TestCoarseStageIsLexical:
    def test_coarse_filter_never_touches_embeddings(self, monkeypatch):
        """粗筛必须零 API 调用 —— 这是"两级不同信号"的核心。"""

        def _boom(self):
            raise AssertionError("coarse_filter 不应该加载 embedding 模型")

        monkeypatch.setattr(Reranker, "_get_embedding_model", _boom)
        reranker = Reranker()
        items = [_item("A", "LangGraph 反射循环"), _item("B", "Python 基础")]
        selected, query_vec = reranker.coarse_filter("LangGraph", items, k=10)
        assert len(selected) == 2
        assert query_vec is None
        # 词法命中的排前面
        assert selected[0]["source_id"] == "A"

    def test_coarse_does_not_drop_non_matching(self):
        """词法不命中的候选也要保留（按原顺序补在后面），不能提前丢召回。"""
        reranker = Reranker(use_embedding_rerank=False)
        items = [_item("A", "完全无关"), _item("B", "LangGraph 教程")]
        selected, _ = reranker.coarse_filter("LangGraph", items, k=10)
        assert {item["source_id"] for item in selected} == {"A", "B"}


class TestTwoStagesDisagree:
    def test_fine_stage_can_reorder_coarse_stage(self, monkeypatch):
        """构造"词法命中但语义不相关" vs "词法不命中但语义相关"，验证两级结论不同。"""
        mapping = {
            "graph": [0.0, 1.0],  # query 向量
            "graph graph": [1.0, 0.0],  # 词法强、语义正交
            "topology": [0.0, 1.0],  # 词法零命中、语义完全一致
        }
        fake = _FakeEmbeddings(mapping)
        monkeypatch.setattr(Reranker, "_get_embedding_model", lambda self: fake)

        reranker = Reranker()
        items = [_item("A", "graph graph"), _item("B", "topology")]

        # 第一级：BM25 把 A 排前面
        coarse, _ = reranker.coarse_filter("graph", items, k=10)
        assert coarse[0]["source_id"] == "A"

        # 第二级：语义把 B 排前面
        results, stats = reranker.rerank("graph", items, coarse_k=10, fine_k=1)
        assert results[0]["source_id"] == "B"
        assert stats["method"] == "bm25+embedding"
        assert stats["coarse_stage"] == "bm25"
        assert stats["fine_stage"] == "embedding"
        assert stats["coarse_kept"] == 2
        assert stats["total_before"] == 2
        assert stats["total_after"] == 1

    def test_cross_encoder_backend_is_used_when_enabled(self, monkeypatch):
        """启用 cross-encoder 时走它，且 method 能如实反映。"""

        class _FakeCrossEncoder:
            def predict(self, pairs):
                # 第二个候选分更高
                return [0.1, 0.9][: len(pairs)]

        fake_ce = _FakeCrossEncoder()
        monkeypatch.setattr(Reranker, "_get_cross_encoder", lambda self: fake_ce)

        reranker = Reranker(use_cross_encoder=True)
        items = [_item("A", "graph graph"), _item("B", "topology")]
        results, stats = reranker.rerank("graph", items, coarse_k=10, fine_k=1)
        assert results[0]["source_id"] == "B"
        assert stats["method"] == "bm25+cross_encoder"
        assert stats["fine_stage"] == "cross_encoder"

    def test_fine_rank_degrades_without_backend(self, monkeypatch):
        """没有任何精排后端时，保留粗筛顺序并把 fine_stage 标成 none。"""

        def _boom(self):
            raise AssertionError("不应加载 embedding 模型")

        monkeypatch.setattr(Reranker, "_get_embedding_model", _boom)
        reranker = Reranker(use_embedding_rerank=False)
        items = [_item(str(i), f"LangGraph 文档 {i}") for i in range(5)]
        results, stats = reranker.rerank("LangGraph", items, coarse_k=10, fine_k=2)
        assert stats["method"] == "bm25_only"
        assert stats["fine_stage"] == "none"
        assert len(results) == 2

    def test_no_internal_fields_leak(self, monkeypatch):
        fake = _FakeEmbeddings({"q": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]})
        monkeypatch.setattr(Reranker, "_get_embedding_model", lambda self: fake)
        reranker = Reranker()
        items = [_item("A", "a"), _item("B", "b")]
        results, _ = reranker.rerank("q", items, coarse_k=10, fine_k=2)
        for item in results:
            assert not [key for key in item if key.startswith("_rerank_")]
