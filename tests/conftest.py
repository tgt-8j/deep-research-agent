"""pytest configuration: redirect tmp_path to a project-local directory.

On Windows, pytest default to %TEMP%\\pytest-of-<user>, which can fail
with PermissionError if that directory was created by another process
(e.g. Codex sandbox). Redirecting to tests/_tmp avoids the issue.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def pytest_configure(config):
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(base)
