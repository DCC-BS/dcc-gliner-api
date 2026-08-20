"""Pydantic request models for the GLiNER2 API, grouped by capability."""

from .classification import BatchClassifyTextRequest, ClassifyTextRequest
from .common import (
    BatchOptions,
    EntityTypes,
    ExtractOptions,
    FieldSpec,
    RelationTypes,
    ResultOptions,
    TasksSpec,
    ValidatorSpec,
)
from .entities import BatchExtractEntitiesRequest, ExtractEntitiesRequest
from .relations import BatchExtractRelationsRequest, ExtractRelationsRequest
from .schema import BatchExtractRequest, ExtractRequest
from .structured import BatchExtractJsonRequest, ExtractJsonRequest

__all__ = [
    "BatchClassifyTextRequest",
    "BatchExtractEntitiesRequest",
    "BatchExtractJsonRequest",
    "BatchExtractRelationsRequest",
    "BatchExtractRequest",
    "BatchOptions",
    "ClassifyTextRequest",
    "EntityTypes",
    "ExtractEntitiesRequest",
    "ExtractJsonRequest",
    "ExtractOptions",
    "ExtractRelationsRequest",
    "ExtractRequest",
    "FieldSpec",
    "RelationTypes",
    "ResultOptions",
    "TasksSpec",
    "ValidatorSpec",
]
