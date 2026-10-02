"""结构化日志配置。"""
from __future__ import annotations

import logging
import sys


class JsonFormatter(logging.Formatter):
    """输出 JSON 格式的日志，便于 ELK/Loki 聚合。

    字段说明：
      - timestamp: ISO 8601 时间戳
      - level: 日志级别
      - logger: 日志器名称
      - message: 日志消息
      - module / function / line: 源码位置
      - trace_id: 请求追踪 ID（通过 extra 传入）
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)
        return super().format(record)


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志：JSON 格式输出到 stdout。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # 避免重复添加 handler
    if not root.handlers:
        root.addHandler(handler)

    # 降低第三方库日志噪音
    for lib in ("uvicorn", "httpx", "openai", "langchain"):
        logging.getLogger(lib).setLevel(logging.WARNING)
