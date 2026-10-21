"""pytest 配置：自己实现 ``tmp_path``，绕开 Windows 上 0o700 目录的坑。

问题链路（实测确认）：
1. pytest 内部的 ``make_numbered_dir`` 固定用 ``mode=0o700`` 建临时目录；
2. 在 Windows + 部分沙箱/加固环境下，``os.mkdir(path, 0o700)`` 建出来的目录
   连**创建者自己都列不出来**（``PermissionError``，3 行可复现）；
3. 于是所有用到 ``tmp_path`` 的用例在 setup 阶段直接 error。上一版
   ``tests/_tmp`` 就是这么变坏、并且再也删不掉的根因。

本项目只有 ``tmp_path`` 一处用到 pytest 的临时目录机制（34 处引用，
没有用 ``tmp_path_factory`` / ``tmpdir``），所以这里直接替换掉该 fixture：
用默认权限建目录、每次会话一个全新根目录、会话结束清理。
效果：在任何机器上都稳定，且不再需要仓库内的 ``tests/_tmp``。
"""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 仓库里可能残留历史版本的 tests/_tmp（0o700 留下的不可读目录），
# 明确告诉 pytest 不要往里收集。
collect_ignore = ["_tmp"]
collect_ignore_glob = ["_tmp/*"]


def _pick_root() -> Path:
    """选一个可用的临时根目录（默认权限创建，不触发 0o700 问题）。"""
    candidates = [
        PROJECT_ROOT / ".pytest_tmp" / uuid.uuid4().hex,
        Path(tempfile.gettempdir()) / f"deep-research-pytest-{uuid.uuid4().hex}",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            (candidate / ".write_probe").write_text("ok", encoding="utf-8")
            return candidate
        except OSError:
            continue
    # 都不可用时退回系统临时目录（行为与 pytest 默认一致）
    return Path(tempfile.mkdtemp(prefix="deep-research-pytest-"))


@pytest.fixture(scope="session")
def _tmp_root():
    root = _pick_root()
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def tmp_path(request, _tmp_root) -> Path:
    """每个用例一个独立且空的临时目录（替代 pytest 自带实现）。"""
    safe_name = re.sub(r"[^\w.\-]", "_", request.node.nodeid)[:80]
    path = _tmp_root / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path
