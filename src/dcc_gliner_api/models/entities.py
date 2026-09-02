"""Request models for entity extraction endpoints."""

from pydantic import BaseModel, Field

from dcc_gliner_api.models.common import BatchOptions, BatchProgress, EntityTypes, ResultOptions


class Entity(BaseModel):
    text: str
    start: int
    end: int
    confidence: float


class ExtractEntitiesOptions(ResultOptions):
    """Everything about an extraction except the text itself.

    Split out so the text can arrive separately as an uploaded file without
    the JSON callers having to change.
    """

    entity_types: EntityTypes = Field(..., description="List of labels, or dict label -> description")


class ExtractEntitiesRequest(ExtractEntitiesOptions):
    text: str


class BatchExtractEntitiesRequest(ResultOptions, BatchOptions):
    texts: list[str]
    entity_types: EntityTypes


class ExtractEntitiesResponse(BaseModel):
    entities: dict[str, list[Entity]]


class ExtractEntitiesBatchResponse(ExtractEntitiesResponse):
    progress: BatchProgress
