"""Rerank 重排序模块：对检索结果进行粗筛+精排两级优化。

设计思路：
- 粗筛（Embedding 向量相似度）：保证召回率，从大量候选中快速筛选 Top-N
- 精排（Cross-Encoder 重排序）：保证准确率，对精选候选进行精细打分
- 两级策略平衡速度和质量，适合生产环境

使用方式：
    from src.retrieval.reranker import Reranker, rerank_evidence
    reranker = Reranker()
    results = reranker.rerank(query, evidence_items, coarse_k=30, fine_k=10)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("research.retrieval.reranker")


class Reranker:
    """Rerank 重排序器。

    属性：
        use_embedding_rerank: 是否使用 Embedding 重排序（备选方案）
        embedding_model: Embedding 模型实例
    """

    def __init__(self, use_embedding_rerank: bool = True):
        """初始化 Reranker。

        Args:
            use_embedding_rerank: 是否启用 Embedding 重排序（备用方案）
        """
        self.use_embedding_rerank = use_embedding_rerank
        self._embedding_model: Any | None = None

    def _get_embedding_model(self):
        """懒加载 Embedding 模型。"""
        if self._embedding_model is None:
            try:
                import os

                from langchain_community.embeddings import DashScopeEmbeddings

                api_key = os.environ.get("DASHSCOPE_API_KEY", "")
                if api_key:
                    self._embedding_model = DashScopeEmbeddings(
                        model="text-embedding-v1",
                        dashscope_api_key=api_key,
                    )
                    logger.info("[reranker] Embedding 模型加载成功")
                else:
                    logger.warning(
                        "[reranker] DASHSCOPE_API_KEY 未设置，禁用 Embedding 重排序"
                    )
                    self.use_embedding_rerank = False
            except ImportError:
                logger.warning(
                    "[reranker] langchain_community.embeddings 不可用，禁用 Embedding 重排序"
                )
                self.use_embedding_rerank = False
        return self._embedding_model

    def _embed_text(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        model = self._get_embedding_model()
        if model is None:
            raise RuntimeError("Embedding 模型不可用")
        return model.embed_documents(texts)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度。

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度 [0, 1]
        """
        a = np.array(vec1, dtype=np.float64)
        b = np.array(vec2, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def coarse_filter(
        self,
        query: str,
        evidence_items: list[dict],
        k: int = 30,
    ) -> list[dict]:
        """粗筛：用 Embedding 向量相似度从大量候选中筛选 Top-K。

        Args:
            query: 查询词
            evidence_items: 证据列表，每项包含 source_id, title, snippet 等
            k: 筛选数量

        Returns:
            筛选后的证据列表，按相似度降序
        """
        if not evidence_items:
            return []

        # 准备文本用于 embedding
        texts = [
            f"{item.get('title', '')} {item.get('snippet', '')}"
            for item in evidence_items
        ]

        # 生成查询和文档向量
        try:
            query_vec = self._embed_text([query])[0]
            doc_vectors = self._embed_text(texts)
        except Exception as e:
            logger.warning("[coarse_filter] Embedding 失败，跳过粗筛: %s", e)
            return evidence_items[:k]

        # 计算相似度并排序
        scored_items = []
        for i, item in enumerate(evidence_items):
            sim = self._cosine_similarity(query_vec, doc_vectors[i])
            item_copy = dict(item)
            item_copy["_rerank_coarse_score"] = sim
            scored_items.append(item_copy)

        # 按相似度降序排序，取 Top-K
        scored_items.sort(key=lambda x: x["_rerank_coarse_score"], reverse=True)
        return scored_items[:k]

    def fine_rank(
        self,
        query: str,
        evidence_items: list[dict],
        k: int = 10,
    ) -> list[dict]:
        """精排：用 Embedding 重排序对候选进行精细打分。

        注意：由于没有安装 cross-encoder，这里使用 Embedding 余弦相似度作为精排依据。
        实际生产环境可替换为 BGE-Reranker 等 cross-encoder 模型。

        Args:
            query: 查询词
            evidence_items: 已粗筛的证据列表
            k: 返回数量

        Returns:
            精排后的证据列表，按相关性分数降序
        """
        if not evidence_items:
            return []

        # 使用 Embedding 余弦相似度作为精排分数
        texts = [
            f"{item.get('title', '')} {item.get('snippet', '')}"
            for item in evidence_items
        ]

        try:
            query_vec = self._embed_text([query])[0]
            doc_vectors = self._embed_text(texts)
        except Exception as e:
            logger.warning("[fine_rank] Embedding 失败，使用原始顺序: %s", e)
            return evidence_items[:k]

        # 计算精排分数
        scored_items = []
        for i, item in enumerate(evidence_items):
            sim = self._cosine_similarity(query_vec, doc_vectors[i])
            item_copy = dict(item)
            item_copy["_rerank_fine_score"] = sim
            scored_items.append(item_copy)

        # 按精排分数降序排序，取 Top-K
        scored_items.sort(key=lambda x: x["_rerank_fine_score"], reverse=True)

        # 清理内部字段
        result = []
        for item in scored_items[:k]:
            cleaned = {k: v for k, v in item.items() if not k.startswith("_rerank_")}
            result.append(cleaned)

        return result

    def rerank(
        self,
        query: str,
        evidence_items: list[dict],
        coarse_k: int = 30,
        fine_k: int = 10,
    ) -> tuple[list[dict], dict]:
        """两级重排序：粗筛 + 精排。

        Args:
            query: 查询词
            evidence_items: 所有检索结果
            coarse_k: 粗筛数量
            fine_k: 精排数量

        Returns:
            (重排序后的证据列表, 重排序统计信息)
        """
        if not evidence_items:
            return [], {"coarse_filtered": 0, "fine_ranked": 0, "method": "none"}

        total_before = len(evidence_items)
        stats = {"coarse_filtered": total_before, "fine_ranked": 0, "method": ""}

        # 检查是否有可用的重排序方法
        if not self.use_embedding_rerank:
            logger.info("[rerank] Embedding 重排序不可用，返回原始顺序")
            stats["method"] = "none"
            stats["total_before"] = total_before
            stats["total_after"] = min(len(evidence_items), fine_k)
            return evidence_items[:fine_k], stats

        # 第一步：粗筛
        logger.info("[rerank] 开始粗筛，候选数=%d, 目标数=%d", total_before, coarse_k)
        coarse_results = self.coarse_filter(query, evidence_items, k=coarse_k)
        stats["coarse_filtered"] = len(coarse_results)

        if len(coarse_results) <= fine_k:
            # 粗筛结果已少于精排数量，直接返回
            logger.info(
                "[rerank] 粗筛结果 %d <= 精排目标 %d，跳过精排",
                len(coarse_results),
                fine_k,
            )
            stats["method"] = "coarse_only"
            stats["fine_ranked"] = len(coarse_results)
            return coarse_results, stats

        # 第二步：精排
        logger.info(
            "[rerank] 开始精排，候选数=%d, 目标数=%d", len(coarse_results), fine_k
        )
        fine_results = self.fine_rank(query, coarse_results, k=fine_k)
        stats["method"] = "coarse+fine"
        stats["fine_ranked"] = len(fine_results)

        logger.info(
            "[rerank] 重排序完成：原始=%d -> 粗筛=%d -> 精排=%d",
            total_before,
            stats["coarse_filtered"],
            stats["fine_ranked"],
        )

        return fine_results, stats


def rerank_evidence(
    query: str,
    evidence_items: list[dict],
    coarse_k: int = 30,
    fine_k: int = 10,
    reranker: Reranker | None = None,
) -> tuple[list[dict], dict]:
    """便捷函数：对证据进行重排序。

    Args:
        query: 查询词
        evidence_items: 证据列表
        coarse_k: 粗筛数量
        fine_k: 精排数量
        reranker: Reranker 实例（可选，默认创建新实例）

    Returns:
        (重排序后的证据列表, 统计信息)
    """
    if reranker is None:
        reranker = Reranker()
    return reranker.rerank(query, evidence_items, coarse_k=coarse_k, fine_k=fine_k)


__all__ = ["Reranker", "rerank_evidence"]
