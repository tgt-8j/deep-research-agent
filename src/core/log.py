"""日志与请求追踪系统。

参考 shopkeeper-agent 的 core/log.py + Paper-Agent 的 logging_utils.py：
- 使用 ContextVar 注入 request_id，实现并发请求的日志隔离
- 提供结构化的 logger，所有节点复用同一套日志配置
- 支持 DEBUG/INFO/WARNING/ERROR 四级日志
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

# 每个请求的独立 request_id，用于日志串联排查
_request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


def set_request_id(request_id: Optional[str] = None) -> str:
    """为当前请求设置 request_id，返回实际使用的 ID。"""
    rid = request_id or str(uuid.uuid4())[:8]
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    """获取当前请求的 request_id。"""
    return _request_id_var.get()


class RequestLogger:
    """带 request_id 的结构化日志器。

    用法：
        logger = RequestLogger(__name__)
        logger.info("节点开始", extra={"query": "xxx"})
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _fmt(self, msg: str, **extra: Any) -> str:
        """格式化消息，追加 request_id。"""
        rid = get_request_id()
        extra_str = f" | rid={rid}"
        if extra:
            extra_str += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        return f"[{self._logger.name}]{extra_str} | {msg}"

    def debug(self, msg: str, **extra: Any) -> None:
        self._logger.debug(self._fmt(msg, **extra))

    def info(self, msg: str, **extra: Any) -> None:
        self._logger.info(self._fmt(msg, **extra))

    def warning(self, msg: str, **extra: Any) -> None:
        self._logger.warning(self._fmt(msg, **extra))

    def error(self, msg: str, **extra: Any) -> None:
        self._logger.error(self._fmt(msg, **extra), exc_info=extra.pop("exc_info", None))

    def exception(self, msg: str, **extra: Any) -> None:
        self._logger.exception(self._fmt(msg, **extra))


# 模块级便捷函数
def get_logger(name: str) -> RequestLogger:
    """获取带 request_id 的日志器。"""
    return RequestLogger(name)


def setup_logging(level: str = "INFO") -> None:
    """初始化日志配置。"""
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
