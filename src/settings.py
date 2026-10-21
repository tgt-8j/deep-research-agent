"""项目级路径解析：项目根 / .env / config.json 的唯一真源。

背景：改造前有 4 处各自拼 ``Path(__file__).resolve().parents[n]``，
其中 2 处偏移量算错（指向了项目根的上一层），导致根目录的 ``.env``
与 ``config.json`` 读不到，只能靠导入 pymilvus 时的副作用把环境变量蒙进来。

规则：任何模块需要项目路径，一律从这里导入，不要再自己算 parents。
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("research.settings")

# src/settings.py -> parents[0] = src/, parents[1] = 项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_PATH: Path = PROJECT_ROOT / ".env"
CONFIG_PATH: Path = PROJECT_ROOT / "config.json"
DATA_DIR: Path = PROJECT_ROOT / "data"


def load_env(override: bool = False) -> bool:
    """加载项目根目录的 .env。

    Returns:
        找到并加载返回 True；文件不存在返回 False（不抛异常）。
    """
    if not ENV_PATH.exists():
        logger.debug("未找到 .env: %s", ENV_PATH)
        return False
    load_dotenv(ENV_PATH, override=override)
    return True


# 导入即加载：保证任何 src.* 模块被引用时环境变量已就绪。
# load_dotenv 不覆盖已存在的进程环境变量，重复调用是幂等的。
load_env()

__all__ = [
    "CONFIG_PATH",
    "DATA_DIR",
    "ENV_PATH",
    "PROJECT_ROOT",
    "load_env",
]
