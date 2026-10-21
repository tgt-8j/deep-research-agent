"""Prometheus 指标收集模块。

历史沿革：本模块原先只是 ``app.metrics`` 的重导出壳，导致 ``src/`` 下
12 个节点全部反向依赖 legacy 的 ``app`` 包。这里把实现搬进来，
``src/`` 从此不再依赖 ``app``。

对外接口与旧模块保持一致，故调用方可直接改导入路径、无需改用法：
- ``track_node(name)``：异步节点装饰器，自动记录调用次数与耗时
- ``node_timer(name)``：上下文管理器形式
- ``record_error(phase)``
- ``create_metrics_endpoint()``：返回 FastAPI 路由处理器
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, ContextManager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

logger = logging.getLogger("research.metrics")

# 独立 registry：避免与进程内其他库的默认 registry 冲突
_registry = CollectorRegistry()

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

NODE_DURATION_HISTOGRAM = Histogram(
    "node_duration_seconds",
    "各研究节点处理耗时",
    labelnames=["node"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=_registry,
)


def track_node(node_name: str) -> Callable:
    """装饰器：记录节点的调用次数、耗时与错误数。

    只适用于 ``async def`` 节点（本项目所有节点都是异步的）。
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            NODE_CALL_COUNTER.labels(node=node_name).inc()
            try:
                return await func(*args, **kwargs)
            except Exception:
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
    """上下文管理器形式的耗时统计。"""
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
    """返回 FastAPI 路由处理器，暴露 /metrics 供 Prometheus 抓取。"""

    async def metrics() -> Response:
        return Response(
            content=generate_latest(_registry).decode("utf-8"),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return metrics


__all__ = [
    "ERROR_COUNTER",
    "NODE_CALL_COUNTER",
    "NODE_DURATION_HISTOGRAM",
    "create_metrics_endpoint",
    "node_timer",
    "record_error",
    "track_node",
]
