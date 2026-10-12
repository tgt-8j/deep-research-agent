"""Backend service layer.

NOTE: The old monolithic workflow_service is deprecated in favor of
the new src/services/workflow.py (dependency-injected, async-native).
"""

from functools import lru_cache

from ..config import AppSettings
from src.services.workflow import WorkflowService


@lru_cache(maxsize=1)
def get_workflow_service():
    settings = AppSettings()
    from src.config import AppConfig
    app_config = AppConfig.from_file(settings.config_path)
    return WorkflowService(config=app_config)


__all__ = ["WorkflowService", "get_workflow_service", "task_queue"]
