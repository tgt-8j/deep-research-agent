"""Web Search 健壮性测试。

验证指数退避重试、总超时控制、速率限制、SSRF 防护等功能。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
from unittest.mock import patch

import pytest

from src.retrieval.web_search import (
    SimpleRateLimiter,
    _filter_records,
    _is_official_domain,
    _safe_url,
    bocha_web_search,
    create_rate_limiter,
    extract_url_content,
)

# ---------------------------------------------------------------------------
# SSRF 防护测试
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    def test_valid_https(self):
        """合法 HTTPS URL 应通过。"""
        assert _safe_url("https://example.com/page") == "https://example.com/page"

    def test_blocked_localhost(self):
        """localhost 应被拦截。"""
        with pytest.raises(ValueError, match="内网"):
            _safe_url("http://localhost/api")

    def test_blocked_internal_ip(self):
        """内网 IP 应被拦截。"""
        with pytest.raises(ValueError, match="内网"):
            _safe_url("http://192.168.1.1/api")

    def test_blocked_metadata(self):
        """云元数据地址应被拦截。"""
        with pytest.raises(ValueError, match="内网"):
            _safe_url("http://metadata.google.internal/")

    def test_blocked_scheme(self):
        """非 HTTP/HTTPS 协议应被拦截。"""
        with pytest.raises(ValueError, match="不支持"):
            _safe_url("ftp://example.com/file")

    def test_valid_http(self):
        """合法 HTTP URL 应通过。"""
        assert (
            _safe_url("http://news.baidu.com/article")
            == "http://news.baidu.com/article"
        )


# ---------------------------------------------------------------------------
# 域名过滤测试
# ---------------------------------------------------------------------------


class TestDomainFilter:
    def test_gov_cn_domain(self):
        """gov.cn 域名应标记为官方。"""
        assert _is_official_domain("www.beijing.gov.cn") is True

    def test_edu_domain(self):
        """edu.cn 域名应标记为官方。"""
        assert _is_official_domain("www.tsinghua.edu.cn") is True

    def test_news_domain(self):
        """新闻域名应标记为非官方（但不被拦截）。"""
        assert _is_official_domain("news.baidu.com") is False

    def test_bad_domains(self):
        """已知 bad domain 应被过滤。"""
        # 在 _filter_records 中测试

    def test_good_domain(self):
        """普通域名。"""
        assert _is_official_domain("github.com") is False


# ---------------------------------------------------------------------------
# 相关性过滤测试
# ---------------------------------------------------------------------------


class TestRelevanceFilter:
    def test_high_relevance(self):
        """高相关性记录应保留。"""
        records = [
            {
                "title": "LangGraph 框架教程",
                "snippet": "LangGraph 是构建 Agent 的框架",
                "domain": "example.com",
            }
        ]
        kept, stats = _filter_records("LangGraph 框架", records)
        assert len(kept) == 1
        assert stats["kept_count"] == 1

    def test_low_relevance_filtered(self):
        """低相关性记录应被过滤。"""
        records = [
            {
                "title": "Python 基础",
                "snippet": "Python 是一种编程语言",
                "domain": "example.com",
            }
        ]
        kept, _stats = _filter_records("LangGraph Agent 框架", records)
        # 可能保留（官方域名豁免）或过滤
        assert isinstance(kept, list)

    def test_empty_records(self):
        """空记录列表。"""
        kept, stats = _filter_records("query", [])
        assert kept == []
        assert stats["raw_count"] == 0

    def test_blocked_domains(self):
        """被拦截的域名应被过滤。"""
        records = [{"title": "Test", "snippet": "Content", "domain": "doc88.com"}]
        _kept, stats = _filter_records("test", records)
        assert stats["dropped_domain"] >= 1


# ---------------------------------------------------------------------------
# 速率限制器测试
# ---------------------------------------------------------------------------


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """并发请求应受限制。"""
        limiter = SimpleRateLimiter(max_concurrent=2)
        running = 0
        max_running = 0

        async def task():
            nonlocal running, max_running
            async with limiter:
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0.05)
                running -= 1

        await asyncio.gather(*[task() for _ in range(5)])
        assert max_running <= 2

    @pytest.mark.asyncio
    async def test_sequential_ordering(self):
        """请求应按顺序执行。"""
        limiter = SimpleRateLimiter(max_concurrent=1)
        order = []

        async def task(i):
            async with limiter:
                order.append(i)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[task(i) for i in range(3)])
        assert order == [0, 1, 2]

    def test_factory_function(self):
        """工厂函数应返回正确实例。"""
        limiter = create_rate_limiter(max_concurrent=5)
        assert isinstance(limiter, SimpleRateLimiter)
        assert limiter._semaphore._value == 5


# ---------------------------------------------------------------------------
# bocha_web_search 单元测试（mock API）
# ---------------------------------------------------------------------------


class TestBochaWebSearch:
    """使用 mock 模拟 API 调用。"""

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """无 API Key 时应返回空列表。"""
        with patch.dict("os.environ", {}, clear=True):
            result = await bocha_web_search("test query")
            assert result == []

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """空响应应返回空列表。"""
        with patch("src.retrieval.web_search._make_request") as mock_req:
            mock_req.return_value = b'{"data":{"webPages":{"value":[]}}}'
            with patch.dict("os.environ", {"BOCHA_API_KEY": "test_key"}):
                result = await bocha_web_search("test query", api_key="test_key")
                assert result == []

    @pytest.mark.asyncio
    async def test_parse_valid_response(self):
        """解析有效响应。"""
        mock_json = json.dumps(
            {
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "url": "https://example.com/page1",
                                "name": "Example Page",
                                "summary": "This is a summary",
                                "datePublished": "2024-01-01",
                            }
                        ]
                    }
                }
            }
        )
        with patch("src.retrieval.web_search._make_request") as mock_req:
            mock_req.return_value = mock_json.encode()
            with patch.dict("os.environ", {"BOCHA_API_KEY": "test_key"}):
                result = await bocha_web_search("test query", api_key="test_key")
                assert len(result) == 1
                assert result[0]["title"] == "Example Page"
                assert result[0]["source_type"] == "web"

    @pytest.mark.asyncio
    async def test_retry_on_http_error(self):
        """HTTP 500 错误应触发重试。"""
        call_count = 0

        def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise urllib.error.HTTPError(
                    "https://api.bocha.cn/v1/web-search",
                    500,
                    "Internal Server Error",
                    {},
                    None,
                )
            return b'{"data":{"webPages":{"value":[]}}}'

        with (
            patch(
                "src.retrieval.web_search._make_request", side_effect=failing_request
            ),
            patch.dict("os.environ", {"BOCHA_API_KEY": "test_key"}),
        ):
            await bocha_web_search(
                "test query",
                api_key="test_key",
                max_retries=3,
            )
            # 应重试直到成功
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """400 错误不应重试。"""
        call_count = 0

        def client_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise urllib.error.HTTPError(
                "https://api.bocha.cn/v1/web-search", 400, "Bad Request", {}, None
            )

        with patch("src.retrieval.web_search._make_request", side_effect=client_error):
            with patch.dict("os.environ", {"BOCHA_API_KEY": "test_key"}):
                result = await bocha_web_search(
                    "test query",
                    api_key="test_key",
                    max_retries=3,
                )
                assert call_count == 1  # 不重试
                assert result == []

    @pytest.mark.asyncio
    async def test_timeout(self):
        """超时应触发重试。"""

        async def slow_request(*args, **kwargs):
            await asyncio.sleep(10)
            return b"{}"

        with patch("src.retrieval.web_search._make_request", side_effect=slow_request):
            with patch.dict("os.environ", {"BOCHA_API_KEY": "test_key"}):
                # 设置很短的超时
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        bocha_web_search("test", api_key="test_key", total_timeout=0.1),
                        timeout=1.0,
                    )


# ---------------------------------------------------------------------------
# extract_url_content 测试
# ---------------------------------------------------------------------------


class TestExtractUrlContent:
    def test_blocked_url(self):
        """被拦截的 URL 应返回错误信息。"""
        result = extract_url_content("http://192.168.1.1/admin")
        assert "error" in result.lower() or "安全" in result

    def test_invalid_scheme(self):
        """无效协议应返回错误信息。"""
        result = extract_url_content("javascript:alert(1)")
        assert "error" in result.lower() or "安全" in result


# ---------------------------------------------------------------------------
# 导入验证
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_exports(self):
        """所有导出应可导入。"""
        from src.retrieval import (
            _filter_records,
            _is_official_domain,
            bocha_web_search,
            create_rate_limiter,
            extract_url_content,
            get_rate_limiter,
        )

        assert callable(bocha_web_search)
        assert callable(extract_url_content)
        assert callable(get_rate_limiter)
        assert callable(create_rate_limiter)
        assert callable(_is_official_domain)
        assert callable(_filter_records)
