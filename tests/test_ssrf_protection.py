"""SSRF 防护测试：_safe_url 函数"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from mult_agents.tools import _safe_url


class TestSafeUrl:
    """合法 URL 应通过校验。"""

    @pytest.mark.parametrize("url", [
        "https://example.com/page",
        "http://example.com/page",
        "https://www.google.com/search?q=test",
        "http://news.ycombinator.com/item?id=12345",
    ])
    def test_valid_urls(self, url):
        result = _safe_url(url)
        assert result == url

    def test_returns_original_url(self):
        url = "https://example.com/path?query=value"
        assert _safe_url(url) == url


class TestBlockedSchemes:
    """禁止的协议应被拒绝。"""

    @pytest.mark.parametrize("url", [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "gopher://example.com",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "",
    ])
    def test_blocked_schemes(self, url):
        with pytest.raises(ValueError, match="不支持的协议"):
            _safe_url(url)


class TestBlockedHosts:
    """内网地址应被拒绝。"""

    @pytest.mark.parametrize("url", [
        "http://localhost/api",
        "http://127.0.0.1:8080/",
        "http://0.0.0.0/admin",
        "http://[::1]/api",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.1/admin",
        "http://172.16.0.1:6379/",
        "http://172.31.255.255/",
    ])
    def test_blocked_hosts(self, url):
        with pytest.raises(ValueError, match="禁止访问|内网"):
            _safe_url(url)

    @pytest.mark.parametrize("prefix", [
        "10.", "172.16.", "172.31.", "192.168.", "169.254.", "127."
    ])
    def test_ip_prefix_blocks(self, prefix):
        with pytest.raises(ValueError, match="禁止访问|内网"):
            _safe_url(f"http://{prefix}0.0/{prefix}1")


class TestEmptyHost:
    """缺少主机名的 URL 应被拒绝。"""

    def test_empty_host(self):
        with pytest.raises(ValueError, match="缺少有效主机名"):
            _safe_url("http:///path")

    def test_no_host(self):
        with pytest.raises(ValueError):
            _safe_url(":///path")
