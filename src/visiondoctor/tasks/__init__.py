"""Task-level evidence, execution, and reference adapters."""

from visiondoctor.tasks.adapters import (
    DatasetTaskAdapter,
    PreparedTaskCase,
    get_task_adapter,
    supported_task_capabilities,
)

__all__ = [
    "DatasetTaskAdapter",
    "PreparedTaskCase",
    "get_task_adapter",
    "supported_task_capabilities",
]
