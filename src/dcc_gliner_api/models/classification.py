"""Request models for text classification endpoints."""

from typing import Any

from pydantic import Field

from dcc_gliner_api.models.common import BatchOptions, ExtractOptions, TasksSpec


class ClassifyTextRequest(ExtractOptions):
    text: str
    tasks: TasksSpec = Field(
        ...,
        description='Classification tasks, e.g. {"sentiment": ["positive","negative"]} '
        'or {"aspects": {"labels": ["camera","battery"], "multi_label": true, "cls_threshold": 0.4}}',
    )


class BatchClassifyTextRequest(ExtractOptions, BatchOptions):
    texts: list[str]
    tasks: dict[str, Any]
