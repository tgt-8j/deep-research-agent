"""检索工具模块：封装 Bocha Web Search 和本地知识库检索。

参考 deep_research/app/mult_agents/tools.py，将全局 init 模式改为
显式传入 API Key 的方式，与 Runtime Context 模式保持一致。

改进（Day 9-10）：
- 指数退避重试（最多 3 次，间隔 1s/2s/4s）
- 总超时控制（asyncio.wait_for，默认 60s）
- 请求速率限制（同一进程内并发上限 3）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger("research.retrieval")

# SSRF 防护
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
}
_BLOCKED_IP_PREFIXES = (
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "192.168.",
    "169.254.",
    "127.",
)


# ---------------------------------------------------------------------------
# 速率限制器（线程安全）
# ---------------------------------------------------------------------------


class SimpleRateLimiter:
    """简单令牌桶速率限制器。

    限制同一进程内的并发请求数，避免触发 API 限流。
    """

    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._min_interval = 0.5  # 最小请求间隔（秒）
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """获取请求许可。"""
        async with self._semaphore, self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


# 全局速率限制器（单例）
_rate_limiter: SimpleRateLimiter | None = None


def get_rate_limiter(max_concurrent: int = 3) -> SimpleRateLimiter:
    """获取全局速率限制器实例。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SimpleRateLimiter(max_concurrent=max_concurrent)
    return _rate_limiter


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _safe_url(url: str) -> str:
    """校验 URL 安全性，防止 SSRF 攻击。"""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"不支持的协议: {parsed.scheme!r}，仅允许 http/https")
    host = parsed.hostname or ""
    if host in _BLOCKED_HOSTS or any(host.startswith(p) for p in _BLOCKED_IP_PREFIXES):
        raise ValueError(f"禁止访问内网或保留地址: {host!r}")
    if not host:
        raise ValueError("URL 缺少有效主机名")
    return url


def _is_official_domain(domain: str) -> bool:
    value = domain.lower()
    return (
        value.endswith((".gov.cn", ".gov", ".edu", ".edu.cn"))
        or "gov" in value
        or "official" in value
    )


def _is_bad_web_domain(domain: str) -> bool:
    value = domain.lower()
    blocked = ["datasheet", "bdtic", "doc88", "elecfans", "down"]
    return any(item in value for item in blocked)


def _extract_query_terms(query: str) -> list[str]:
    """提取查询关键词用于相关性评估。"""
    parts = re.findall(r"[一-鿿]{2,}|[A-Za-z0-9_-]{3,}", query.lower())
    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于"}
    return [p for p in parts if p not in stopwords][:12]


def _estimate_relevance(query: str, text: str) -> float:
    """估算文本与查询的相关性分数。"""
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _filter_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    """过滤低质量搜索结果。"""
    kept = []
    stats = {
        "raw_count": len(records),
        "kept_count": 0,
        "dropped_irrelevant": 0,
        "dropped_domain": 0,
    }
    blocked_domains = ["datasheet", "bdtic", "doc88", "elecfans", "down"]

    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        domain = str(record.get("domain", ""))
        if not title and not snippet:
            stats["dropped_domain"] += 1
            continue
        if any(b in domain.lower() for b in blocked_domains):
            stats["dropped_domain"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if relevance < 0.2 and not _is_official_domain(domain):
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------------
# 核心搜索函数（带重试 + 超时）
# ---------------------------------------------------------------------------


def _make_request(payload: dict, key: str, timeout: int = 30) -> bytes:
    """发起 HTTP 请求，返回响应体字节。

    Args:
        payload: 请求体
        key: API Key
        timeout: 单次请求超时（秒）

    Returns:
        响应体字节
    """
    request = urllib.request.Request(
        url="https://api.bocha.cn/v1/web-search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


async def bocha_web_search(
    query: str,
    count: int = 4,
    api_key: str | None = None,
    max_retries: int = 3,
    total_timeout: int = 60,
    rate_limit: SimpleRateLimiter | None = None,
) -> list[dict]:
    """调用 Bocha Web Search API 执行网页搜索（带重试 + 超时 + 速率限制）。

    改进：
    - 指数退避重试：失败后等待 1s/2s/4s 再重试，最多 3 次
    - 总超时控制：整个搜索过程（含重试）不超过 total_timeout 秒
    - 速率限制：同一进程内并发请求上限，避免触发 API 限流

    Args:
        query: 搜索查询词
        count: 返回结果数量
        api_key: Bocha API Key，不传则从环境变量 BOCHA_API_KEY 读取
        max_retries: 最大重试次数
        total_timeout: 总超时时间（秒），包括重试
        rate_limiter: 速率限制器（可选，默认使用全局实例）

    Returns:
        原始搜索结果列表，每条包含 source_id, title, url, snippet, domain 等字段
    """
    key = api_key or os.getenv("BOCHA_API_KEY", "").strip()
    if not key:
        logger.warning(
            "[bocha_web_search] 未配置 BOCHA_API_KEY，跳过搜索 | query=%s", query
        )
        return []

    payload = {
        "query": query,
        "summary": True,
        "freshness": "noLimit",
        "count": count,
    }

    limiter = rate_limit or get_rate_limiter()

    # 使用 asyncio.wait_for 控制总超时
    async def _search_with_retry() -> list[dict]:
        last_error = None
        delay = 1.0  # 初始等待时间

        for attempt in range(max_retries + 1):
            try:
                # 获取速率限制许可
                await limiter.acquire()

                # 执行搜索（异步包装同步操作）
                loop = asyncio.get_event_loop()
                raw_bytes = await asyncio.wait_for(
                    loop.run_in_executor(None, _make_request, payload, key, 30),
                    timeout=total_timeout,
                )

                result = json.loads(raw_bytes.decode("utf-8"))
                return _parse_search_result(result, query)

            except asyncio.TimeoutError:
                last_error = f"总超时 {total_timeout}s"
                logger.warning(
                    "[bocha_web_search] 第 %d 次尝试超时: %s | query=%s",
                    attempt + 1,
                    last_error,
                    query[:50],
                )
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                # 4xx 客户端错误不重试
                if 400 <= e.code < 500 and e.code != 429:
                    logger.error(
                        "[bocha_web_search] HTTP 错误（不重试）: %s | query=%s",
                        last_error,
                        query[:50],
                    )
                    return []
                logger.warning(
                    "[bocha_web_search] HTTP 错误（尝试重试）: %s | query=%s",
                    last_error,
                    query[:50],
                )
            except urllib.error.URLError as e:
                last_error = f"URL 错误: {e.reason}"
                logger.warning(
                    "[bocha_web_search] URL 错误（尝试重试）: %s | query=%s",
                    last_error,
                    query[:50],
                )
            except json.JSONDecodeError as e:
                last_error = f"JSON 解析错误: {e}"
                logger.error(
                    "[bocha_web_search] JSON 解析错误（不重试）: %s",
                    e,
                )
                return []
            except Exception as e:
                last_error = f"未知错误: {type(e).__name__}: {e}"
                logger.warning(
                    "[bocha_web_search] 未知错误（尝试重试）: %s | query=%s",
                    last_error,
                    query[:50],
                )

            # 指数退避重试
            if attempt < max_retries:
                logger.info(
                    "[bocha_web_search] 等待 %.1fs 后重试 %d/%d | query=%s",
                    delay,
                    attempt + 1,
                    max_retries,
                    query[:50],
                )
                await asyncio.sleep(delay)
                delay *= 2  # 下次等待时间翻倍

        logger.error(
            "[bocha_web_search] 重试 %d 次后仍然失败 | query=%s | error=%s",
            max_retries,
            query[:50],
            last_error,
        )
        return []

    return await _search_with_retry()


def _parse_search_result(result: dict, query: str) -> list[dict]:
    """解析搜索结果，提取记录列表。"""
    data = result.get("data", {})
    pages = data.get("webPages", [])
    if isinstance(pages, dict):
        if isinstance(pages.get("value"), list):
            pages = pages.get("value", [])
        elif isinstance(pages.get("items"), list):
            pages = pages.get("items", [])
        else:
            pages = []
    if not isinstance(pages, list):
        return []

    records: list[dict] = []
    for idx, page in enumerate(pages, 1):
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "").strip()
        domain = ""
        if "://" in url:
            domain = url.split("://", 1)[1].split("/", 1)[0]
        title = page.get("name") or f"web_result_{idx}"
        snippet = page.get("summary") or ""
        records.append(
            {
                "source_id": f"WEB-{idx}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "source_type": "web",
                "published_at": page.get("datePublished")
                or page.get("dateLastCrawled")
                or "",
            }
        )
    return records


def extract_url_content(url: str, timeout: int = 10) -> str:
    """抓取 URL 页面内容并提取纯文本。

    改进：支持 timeout 参数。

    Args:
        url: 目标 URL
        timeout: 请求超时（秒）

    Returns:
        前 2000 个字符的纯文本，失败时返回错误信息
    """
    try:
        url = _safe_url(url)
    except ValueError as exc:
        return f"[error] URL 安全检查失败: {exc}"
    try:
        req = urllib.request.Request(
            url=url,
            headers={"User-Agent": "DeepResearchBot/1.0 (educational)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000] if len(text) > 2000 else text
    except Exception as e:
        return f"[error] 无法获取 {url}: {e}"


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def create_rate_limiter(max_concurrent: int = 3) -> SimpleRateLimiter:
    """创建独立的速率限制器实例。

    Args:
        max_concurrent: 最大并发数

    Returns:
        SimpleRateLimiter 实例
    """
    return SimpleRateLimiter(max_concurrent=max_concurrent)


__all__ = [
    "SimpleRateLimiter",
    "_estimate_relevance",
    "_extract_query_terms",
    "_filter_records",
    "_is_official_domain",
    "bocha_web_search",
    "create_rate_limiter",
    "extract_url_content",
    "get_rate_limiter",
]
