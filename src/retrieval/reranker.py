"""Rerank 重排序：词法粗筛 + 语义精排（两级用的是**不同**信号）。

为什么要改：旧实现里粗筛和精排都是对 ``title + snippet`` 算 embedding 余弦，
等于同一个信号算了两遍 —— 精排只会得到和粗筛完全相同的顺序，还多付一次
embedding 费用。现在两级的信号真正不同：

    第一级（粗筛，0 次 API 调用）：BM25 词法相关性，负责"高召回 + 砍数量"
        100 条候选 → Top-30。只做排序和截断，词法不命中的候选也不会被丢弃
        （按原顺序补在后面），避免因为分词问题把好证据提前筛掉。

    第二级（精排，语义相关性）：Top-30 → Top-10
        a) cross-encoder（BGE-Reranker 等，精度最高，需装 sentence-transformers
           并下载模型）
        b) embedding 余弦（默认，无额外依赖）

后端选择：默认走 embedding；设置 ``RERANKER_BACKEND=cross_encoder`` 且安装了
``sentence-transformers`` 时启用 cross-encoder（懒加载，加载失败自动降级）。
模型可用 ``RERANKER_MODEL`` 覆盖，默认 ``BAAI/bge-reranker-base``。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from .hybrid_search import bm25_search

logger = logging.getLogger("research.retrieval.reranker")

_DEFAULT_CE_MODEL = "BAAI/bge-reranker-base"
# DashScope text-embedding-v1 单次最多 25 条，留一点余量
_EMBED_BATCH_SIZE = 16
_CE_BACKEND_ALIASES = {"cross_encoder", "cross-encoder", "ce", "bge", "reranker"}


def _item_text(item: dict) -> str:
    return f"{item.get('title', '')} {item.get('snippet', '')}".strip()


def _strip_internal(item: dict) -> dict:
    return {k: v for k, v in item.items() if not k.startswith("_rerank_")}


class Reranker:
    """两级重排序器：BM25 粗筛 → 语义精排。"""

    def __init__(
        self,
        use_embedding_rerank: bool = True,
        use_cross_encoder: bool | None = None,
    ):
        """
        Args:
            use_embedding_rerank: 是否允许用 Embedding 余弦做精排（关掉即只做粗筛）。
            use_cross_encoder: 是否启用 cross-encoder 精排。
                ``None``（默认）表示读环境变量 ``RERANKER_BACKEND`` 决定。
        """
        self.use_embedding_rerank = use_embedding_rerank
        self._embedding_model: Any | None = None
        self._cross_encoder: Any | None = None
        self._cross_encoder_checked = False

        if use_cross_encoder is None:
            backend = os.getenv("RERANKER_BACKEND", "").strip().lower()
            use_cross_encoder = backend in _CE_BACKEND_ALIASES
        self.use_cross_encoder = bool(use_cross_encoder)

    # ------------------------------------------------------------------
    # 后端懒加载
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """懒加载 Embedding 模型（不可用时自动关闭精排）。"""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from langchain_community.embeddings import DashScopeEmbeddings

            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                logger.warning(
                    "[reranker] DASHSCOPE_API_KEY 未设置，禁用 Embedding 精排"
                )
                self.use_embedding_rerank = False
                return None
            self._embedding_model = DashScopeEmbeddings(
                model="text-embedding-v1",
                dashscope_api_key=api_key,
            )
            logger.info("[reranker] Embedding 模型加载成功")
        except ImportError:
            logger.warning("[reranker] langchain_community 不可用，禁用 Embedding 精排")
            self.use_embedding_rerank = False
        return self._embedding_model

    def _get_cross_encoder(self):
        """懒加载 cross-encoder；任何失败都返回 None（由上层降级）。"""
        if self._cross_encoder_checked:
            return self._cross_encoder
        self._cross_encoder_checked = True
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.info(
                "[reranker] 未安装 sentence-transformers，精排回退到 Embedding 余弦"
            )
            return None
        model_name = os.getenv("RERANKER_MODEL", _DEFAULT_CE_MODEL)
        try:
            self._cross_encoder = CrossEncoder(model_name)
            logger.info("[reranker] cross-encoder 加载完成 | model=%s", model_name)
        except Exception as exc:  # noqa: BLE001 - 模型下载/加载失败都要能降级
            logger.warning("[reranker] cross-encoder 加载失败，回退 Embedding: %s", exc)
            self._cross_encoder = None
        return self._cross_encoder

    # ------------------------------------------------------------------
    # 向量工具
    # ------------------------------------------------------------------

    def _embed_text(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量（按批切分，避免单次请求超限）。"""
        model = self._get_embedding_model()
        if model is None:
            raise RuntimeError("Embedding 模型不可用")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(model.embed_documents(batch))
        return vectors

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """余弦相似度，返回 [-1, 1]。"""
        a = np.asarray(vec1, dtype=np.float64)
        b = np.asarray(vec2, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # 第一级：词法粗筛（BM25，无 API 调用）
    # ------------------------------------------------------------------

    def coarse_filter(
        self,
        query: str,
        evidence_items: list[dict],
        k: int = 30,
    ) -> tuple[list[dict], None]:
        """粗筛：按 BM25 词法相关性排序并截断到 Top-K。

        只改变顺序，不因为词法不命中而丢弃候选：BM25 未命中的条目按原顺序
        补在后面，保证召回不退化（真正的相关性判断交给精排）。

        Returns:
            ``(排序后的前 K 条, None)``。第二个元素为历史签名的占位，
            现在粗筛不产生 query 向量。
        """
        if not evidence_items:
            return [], None

        docs = [
            {"_row": idx, "content": _item_text(item)}
            for idx, item in enumerate(evidence_items)
        ]
        try:
            ranked = bm25_search(query, docs, top_k=len(docs))
        except Exception as exc:  # noqa: BLE001 - 词法不可用时退回原顺序
            logger.warning("[coarse_filter] BM25 计算失败，保持原顺序: %s", exc)
            ranked = []

        order = [doc["_row"] for doc in ranked]
        seen = set(order)
        order.extend(idx for idx in range(len(evidence_items)) if idx not in seen)

        selected = [dict(evidence_items[idx]) for idx in order[:k]]
        logger.info(
            "[coarse_filter] BM25 命中 %d/%d，取 Top-%d",
            len(ranked),
            len(evidence_items),
            len(selected),
        )
        return selected, None

    # ------------------------------------------------------------------
    # 第二级：语义精排（cross-encoder / embedding）
    # ------------------------------------------------------------------

    def fine_rank(
        self,
        query: str,
        evidence_items: list[dict],
        k: int = 10,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        """精排：用语义相关性对粗筛结果排序并截断到 Top-K。"""
        ranked, _backend = self._fine_rank(query, evidence_items, k, query_vec)
        return ranked

    def _fine_rank(
        self,
        query: str,
        evidence_items: list[dict],
        k: int,
        query_vec: list[float] | None = None,
    ) -> tuple[list[dict], str]:
        """返回 ``(结果, 使用的后端)``，后端为 cross_encoder / embedding / none。"""
        if not evidence_items:
            return [], "none"

        cross_encoder = self._get_cross_encoder() if self.use_cross_encoder else None
        if cross_encoder is not None:
            try:
                pairs = [(query, _item_text(item)) for item in evidence_items]
                scores = cross_encoder.predict(pairs)
                scored = [
                    (_strip_internal(item), float(score))
                    for item, score in zip(evidence_items, scores, strict=False)
                ]
                scored.sort(key=lambda pair: pair[1], reverse=True)
                logger.info("[fine_rank] cross-encoder 精排完成 | k=%d", k)
                return [item for item, _ in scored[:k]], "cross_encoder"
            except Exception as exc:  # noqa: BLE001 - 推理失败即降级
                logger.warning("[fine_rank] cross-encoder 推理失败，降级: %s", exc)

        if not self.use_embedding_rerank:
            return [_strip_internal(item) for item in evidence_items[:k]], "none"

        texts = [_item_text(item) for item in evidence_items]
        try:
            if query_vec is None:
                query_vec = self._embed_text([query])[0]
            doc_vectors = self._embed_text(texts)
        except Exception as exc:  # noqa: BLE001 - Embedding 不可用就保持顺序
            logger.warning("[fine_rank] Embedding 失败，保持粗筛顺序: %s", exc)
            return [_strip_internal(item) for item in evidence_items[:k]], "none"

        scored = []
        for item, vector in zip(evidence_items, doc_vectors, strict=False):
            similarity = self._cosine_similarity(query_vec, vector)
            scored.append((_strip_internal(item), similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in scored[:k]], "embedding"

    # ------------------------------------------------------------------
    # 组合：粗筛 → 精排
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        evidence_items: list[dict],
        coarse_k: int = 30,
        fine_k: int = 10,
    ) -> tuple[list[dict], dict]:
        """两级重排序：BM25 粗筛 Top-``coarse_k`` → 语义精排 Top-``fine_k``。

        Returns:
            ``(结果列表, stats)``。stats 里的 ``method`` 取值：
            ``none`` / ``coarse_only`` / ``bm25_only`` /
            ``bm25+embedding`` / ``bm25+cross_encoder``。
        """
        total_before = len(evidence_items)
        if not evidence_items:
            return [], {
                "method": "none",
                "total_before": 0,
                "coarse_stage": "none",
                "coarse_kept": 0,
                "fine_stage": "none",
                "total_after": 0,
            }

        coarse_results, _ = self.coarse_filter(query, evidence_items, k=coarse_k)
        stats = {
            "method": "",
            "total_before": total_before,
            "coarse_stage": "bm25",
            "coarse_kept": len(coarse_results),
            "fine_stage": "none",
            "total_after": 0,
        }

        if len(coarse_results) <= fine_k:
            # 粗筛后已经不超过精排目标，省掉一次精排（也就省掉一次 API 调用）
            results = [_strip_internal(item) for item in coarse_results]
            stats["method"] = "coarse_only"
            logger.info(
                "[rerank] 粗筛结果 %d <= 精排目标 %d，跳过精排",
                len(coarse_results),
                fine_k,
            )
        else:
            results, backend = self._fine_rank(query, coarse_results, k=fine_k)
            stats["fine_stage"] = backend
            stats["method"] = "bm25_only" if backend == "none" else f"bm25+{backend}"

        stats["total_after"] = len(results)
        logger.info(
            "[rerank] %s | 原始=%d -> 粗筛=%d -> 精排=%d",
            stats["method"],
            total_before,
            stats["coarse_kept"],
            stats["total_after"],
        )
        return results, stats


def rerank_evidence(
    query: str,
    evidence_items: list[dict],
    coarse_k: int = 30,
    fine_k: int = 10,
    reranker: Reranker | None = None,
) -> tuple[list[dict], dict]:
    """便捷函数：对证据做两级重排序。"""
    if reranker is None:
        reranker = Reranker()
    return reranker.rerank(query, evidence_items, coarse_k=coarse_k, fine_k=fine_k)


__all__ = ["Reranker", "rerank_evidence"]
