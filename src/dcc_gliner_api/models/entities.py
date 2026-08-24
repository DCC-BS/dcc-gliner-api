"""Request models for entity extraction endpoints."""

from pydantic import BaseModel, Field

from dcc_gliner_api.models.common import BatchOptions, BatchProgress, EntityTypes, ResultOptions


class Entity(BaseModel):
    text: str
    start: int
    end: int
    confidence: float


class ExtractEntitiesRequest(ResultOptions):
    text: str
    entity_types: EntityTypes = Field(..., description="List of labels, or dict label -> description")


class BatchExtractEntitiesRequest(ResultOptions, BatchOptions):
    texts: list[str]
    entity_types: EntityTypes


class ExtractEntitiesResponse(BaseModel):
    entities: dict[str, list[Entity]]


class ExtractEntitiesBatchResponse(ExtractEntitiesResponse):
    progress: BatchProgress
