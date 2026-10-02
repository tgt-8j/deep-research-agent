"""单元测试：retrieval 模块。"""

import pytest
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.retrieval.web_search import (
    _is_official_domain,
    _is_bad_web_domain,
    _safe_url,
)


class TestDomainHelpers:
    """测试域名判断函数。"""

    def test_gov_cn_domain(self):
        assert _is_official_domain("example.gov.cn") == True
        assert _is_official_domain("www.gov.cn") == True

    def test_edu_domain(self):
        assert _is_official_domain("MIT.EDU") == True
        assert _is_official_domain("pku.edu.cn") == True

    def test_news_domain(self):
        assert _is_official_domain("example.news.com") == False

    def test_bad_domains(self):
        assert _is_bad_web_domain("example.datasheet.com") == True
        assert _is_bad_web_domain("doc88.com") == True
        assert _is_bad_web_domain("elecfans.com") == True

    def test_good_domain(self):
        assert _is_bad_web_domain("arxiv.org") == False
        assert _is_bad_web_domain("github.com") == False


class TestSSRFProtection:
    """测试 SSRF 防护。"""

    def test_valid_https(self):
        url = _safe_url("https://example.com/path")
        assert url == "https://example.com/path"

    def test_blocked_localhost(self):
        with pytest.raises(ValueError):
            _safe_url("http://localhost:8080/api")

    def test_blocked_internal_ip(self):
        with pytest.raises(ValueError):
            _safe_url("http://192.168.1.1/admin")

    def test_blocked_metadata(self):
        with pytest.raises(ValueError):
            _safe_url("http://metadata.google.internal/")

    def test_blocked_scheme(self):
        with pytest.raises(ValueError):
            _safe_url("file:///etc/passwd")
