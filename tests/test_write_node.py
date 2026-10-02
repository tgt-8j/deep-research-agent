"""单元测试：write_node 引用校验逻辑。"""

import pytest
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graph.nodes.write_node import (
    _extract_citation_ids,
    _validate_and_fix_citations,
)


class TestCitationExtraction:
    """测试引用 ID 提取。"""

    def test_basic_extraction(self):
        text = "根据 [WEB1_1-1] 的研究，LLM 发展迅速。参考 [LOC1_1-3] 的数据。"
        ids = _extract_citation_ids(text)
        assert "WEB1_1-1" in ids
        assert "LOC1_1-3" in ids

    def test_duplicate_dedup(self):
        """重复引用应去重。"""
        text = "[WEB1_1-1] 和 [WEB1_1-1] 都提到了这一点"
        ids = _extract_citation_ids(text)
        assert ids.count("WEB1_1-1") == 1

    def test_no_citations(self):
        text = "这是一段没有引用的文字"
        ids = _extract_citation_ids(text)
        assert ids == []

    def test_preserve_order(self):
        """保持引用出现的顺序。"""
        text = "[WEB1_1-2] 先提到，然后 [WEB1_1-1] 后提到"
        ids = _extract_citation_ids(text)
        assert ids.index("WEB1_1-2") < ids.index("WEB1_1-1")


class TestCitationValidation:
    """测试引用校验和修复。"""

    def test_valid_citations_kept(self):
        text = "根据 [WEB1_1-1] 和 [LOC1_1-1] 的研究"
        valid = {"WEB1_1-1", "LOC1_1-1"}
        fixed, used = _validate_and_fix_citations(text, valid)
        assert "WEB1_1-1" in fixed
        assert "LOC1_1-1" in fixed
        assert "WEB1_1-1" in used
        assert "LOC1_1-1" in used

    def test_invalid_citations_removed(self):
        # INVALID_ID doesn't match the pattern [A-Z]+\d+_\d+-\d+, so it stays unchanged.
        # Use an ID that matches the regex pattern but is not in the valid set.
        text = "根据 [WEB1_1-1] 和 [WEB9_9-9] 的研究"
        valid = {"WEB1_1-1"}
        fixed, used = _validate_and_fix_citations(text, valid)
        assert "WEB1_1-1" in fixed
        assert "WEB9_9-9" not in fixed
        assert used == ["WEB1_1-1"]

    def test_all_invalid(self):
        text = "根据 [BAD1] 和 [BAD2] 的研究"
        valid = set()
        fixed, used = _validate_and_fix_citations(text, valid)
        assert used == []
