"""本地知识库核心：Milvus 向量检索 + 文本切分。

从 ``app/mult_agents/rag/core.py`` 迁入，原因：``src/`` 不应反向依赖
legacy 的 ``app`` 包。逻辑保持不变，仅做导入与格式整理。

与旧实现的唯一行为差异：``enable_milvus=False`` 时不会再尝试连接
Milvus（旧实现同样如此），``search_records`` 在无集合时返回空列表，
调用方据此自动降级为 BM25-only。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility

logger = logging.getLogger("research.retrieval.rag")

# 优先使用 langchain-milvus 新包（类名为 Milvus，非 MilvusVectorStore）
try:
    from langchain_milvus import Milvus as _MilvusVectorStore

    _MILVUS_BACKEND = "langchain_milvus"
except ImportError:  # pragma: no cover - 取决于安装的依赖组合
    from langchain_community.vectorstores import Milvus as _MilvusVectorStore

    _MILVUS_BACKEND = "langchain_community"


@dataclass(frozen=True)
class RAGConfig:
    """知识库配置。"""

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    collection_name: str = "mult_agent_knowledge"
    embedding_model: str = "text-embedding-v1"
    chunk_size: int = 500
    chunk_overlap: int = 50
    enable_milvus: bool = True


class RAGSystem:
    """封装 Embedding + Milvus 的本地知识库。

    注意：``enable_milvus=True`` 时构造即建连，连接失败会抛异常，
    由 ``create_knowledge_base_client`` 统一降级处理。
    """

    def __init__(self, api_key: str, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.api_key = api_key
        self.embeddings = DashScopeEmbeddings(
            model=self.config.embedding_model,
            dashscope_api_key=self.api_key,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self.vectorstore = None
        if not self.config.enable_milvus:
            logger.info("Milvus 未启用，RAG 仅使用嵌入模型")
            return
        self._connect_to_milvus()
        self.vectorstore = _MilvusVectorStore(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            connection_args={
                "uri": f"http://{self.config.milvus_host}:{self.config.milvus_port}"
            },
            auto_id=True,
        )
        logger.info(
            "RAG backend=%s | collection=%s",
            _MILVUS_BACKEND,
            self.config.collection_name,
        )

    def _connect_to_milvus(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
            )
        except Exception as exc:
            logger.error("连接 Milvus 失败: %s", exc)
            raise

    def search(self, query: str, k: int = 3) -> str:
        """检索并返回人类可读的文本摘要。"""
        try:
            records = self.search_records(query, k=k)
            if not records:
                return "未找到相关信息。"
            lines: list[str] = ["检索到的相关信息："]
            for idx, record in enumerate(records, 1):
                lines.append(f"{idx}. {record['snippet']}")
                lines.append(f"   (来源: {record['doc_id']})")
            return "\n".join(lines)
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            return f"检索过程中发生错误: {exc!s}"

    def search_records(self, query: str, k: int = 5) -> list[dict]:
        """向量检索，返回标准化记录列表。"""
        if self.vectorstore is None:
            return []
        if not utility.has_collection(self.config.collection_name):
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        records: list[dict] = []
        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or "").strip()
            title = Path(source).name if source else f"本地知识片段-{idx}"
            records.append(
                {
                    "source_id": f"LOC-{idx}",
                    "doc_id": source,
                    "title": title,
                    "snippet": doc.page_content,
                    "source_type": "local",
                    "metadata": metadata,
                }
            )
        return records

    def search_records_with_scores(self, query: str, k: int = 5) -> list[dict]:
        """带向量分数的检索，每条结果附 ``_vector_score``。

        Milvus 未启用或集合不存在时返回空列表，调用方据此自动降级。
        """
        if self.vectorstore is None or not self.config.enable_milvus:
            return []
        try:
            if not utility.has_collection(self.config.collection_name):
                return []
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            records: list[dict] = []
            for idx, (doc, score) in enumerate(results, 1):
                metadata = doc.metadata or {}
                source = str(metadata.get("source") or "").strip()
                title = Path(source).name if source else f"本地知识片段-{idx}"
                records.append(
                    {
                        "source_id": f"LOC-{idx}",
                        "doc_id": source,
                        "title": title,
                        "snippet": doc.page_content,
                        "source_type": "local",
                        "metadata": metadata,
                        "_vector_score": score,
                    }
                )
            return records
        except Exception as e:
            logger.warning("[search_records_with_scores] 向量检索失败: %s", e)
            return []

    def add_documents(self, documents: list[Document]) -> int:
        if self.vectorstore is None:
            raise RuntimeError("Milvus 未启用，无法写入文档")
        self.vectorstore.add_documents(documents)
        return len(documents)

    def ingest_text(self, text: str, source: str) -> int:
        docs = self.text_splitter.create_documents(
            [text], metadatas=[{"source": source}]
        )
        return self.add_documents(docs)

    def ingest_paths(self, paths: Iterable[Path]) -> int:
        total = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            total += self.ingest_text(text, source=str(path))
        return total


__all__ = ["RAGConfig", "RAGSystem"]
