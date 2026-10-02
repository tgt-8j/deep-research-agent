"""Prompt 加载工具：从文件加载或从内存获取 prompt 模板。

参考 shopkeeper-agent 的 prompt/prompt_loader.py，支持从文件系统加载 YAML
格式的 prompt 模板，同时保留内存字典作为兜底。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from .templates import PROMPTS as _IN_MEMORY_PROMPTS

_PROMPT_DIR = Path(__file__).parent / "templates"


def load_prompt(key: str) -> str:
    """加载指定 key 的 prompt 模板。

    优先级：
    1. 文件系统中对应的 .md 文件
    2. 内存中的默认模板
    """
    file_path = _PROMPT_DIR / f"{key}.md"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8").strip()
    return _IN_MEMORY_PROMPTS.get(key, "")


def list_prompts() -> list[str]:
    """返回所有可用的 prompt key。"""
    return sorted(set(_IN_MEMORY_PROMPTS.keys()))
