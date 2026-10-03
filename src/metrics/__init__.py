"""Metrics 模块：从 app/metrics 重新导出，供 src/ 下的模块使用。"""

from app.metrics import (  # noqa: F401
    ERROR_COUNTER,
    NODE_CALL_COUNTER,
    NODE_DURATION_HISTOGRAM,
    create_metrics_endpoint,
    record_error,
    track_node,
)
