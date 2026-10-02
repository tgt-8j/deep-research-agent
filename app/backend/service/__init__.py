"""Backend service layer.

NOTE: The old monolithic workflow_service is deprecated in favor of
the new src/services/workflow.py (dependency-injected, async-native).
"""

from functools import lru_cache

from ..config import AppSettings
from .workflow_service_deprecated import WorkflowService  # noqa: F401  # deprecated
from .task_queue import task_queue


@lru_cache(maxsize=1)
def get_workflow_service() -> WorkflowService:
    settings = AppSettings()
    return WorkflowService(config_path=settings.config_path)


__all__ = ["WorkflowService", "get_workflow_service", "task_queue"]
