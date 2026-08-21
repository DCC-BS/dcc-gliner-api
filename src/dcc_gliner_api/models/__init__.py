"""Pydantic request models for the GLiNER2 API, grouped by capability."""

from dcc_gliner_api.models.classification import BatchClassifyTextRequest, ClassifyTextRequest
from dcc_gliner_api.models.common import (
    BatchOptions,
    EntityTypes,
    ExtractOptions,
    FieldSpec,
    RelationTypes,
    ResultOptions,
    TasksSpec,
    ValidatorSpec,
)
from dcc_gliner_api.models.entities import BatchExtractEntitiesRequest, ExtractEntitiesRequest
from dcc_gliner_api.models.relations import BatchExtractRelationsRequest, ExtractRelationsRequest
from dcc_gliner_api.models.schema import BatchExtractRequest, ExtractRequest
from dcc_gliner_api.models.structured import BatchExtractJsonRequest, ExtractJsonRequest

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
