"""Prometheus 指标收集模块。

暴露 /metrics 端点，供 Prometheus 抓取。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ContextManager

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from starlette.responses import Response

logger = logging.getLogger("metrics")

_registry = CollectorRegistry()

# 计数器
NODE_CALL_COUNTER = Counter(
    "node_call_total",
    "各研究节点被调用次数",
    labelnames=["node"],
    registry=_registry,
)

ERROR_COUNTER = Counter(
    "node_error_total",
    "各节点错误计数",
    labelnames=["phase"],
    registry=_registry,
)

# 直方图：节点耗时
NODE_DURATION_HISTOGRAM = Histogram(
    "node_duration_seconds",
    "各研究节点处理耗时",
    labelnames=["node"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=_registry,
)


def track_node(node_name: str) -> Callable:
    """装饰器：自动记录节点调用次数和耗时。"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            NODE_CALL_COUNTER.labels(node=node_name).inc()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                ERROR_COUNTER.labels(phase=node_name).inc()
                raise
            finally:
                NODE_DURATION_HISTOGRAM.labels(node=node_name).observe(
                    time.perf_counter() - start
                )
        return wrapper
    return decorator


@contextmanager
def node_timer(node_name: str) -> ContextManager[None]:
    """上下文管理器形式使用。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        NODE_DURATION_HISTOGRAM.labels(node=node_name).observe(
            time.perf_counter() - start
        )


def record_error(phase: str) -> None:
    """记录某阶段错误。"""
    ERROR_COUNTER.labels(phase=phase).inc()


def create_metrics_endpoint() -> Callable:
    """返回 FastAPI 路由处理器，暴露 /metrics。"""
    async def metrics() -> Response:
        return Response(
            content=generate_latest(_registry).decode("utf-8"),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
    return metrics
