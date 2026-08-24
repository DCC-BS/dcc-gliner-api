"""Request models for entity extraction endpoints."""
from typing import Dict
from markdown_it.common.entities import entities

from pydantic import Field, BaseModel

from dcc_gliner_api.models.common import BatchOptions, EntityTypes, ResultOptions, BatchProgress

class Entity(BaseModel):
    text: str
    start: int
    end: int
    confidence: float

class ExtractEntitiesRequest(ResultOptions):
    text: str
    entity_types: EntityTypes = Field(
        ..., description="List of labels, or dict label -> description"
    )


class BatchExtractEntitiesRequest(ResultOptions, BatchOptions):
    texts: list[str]
    entity_types: EntityTypes

class ExtractEntitiesResponse(BaseModel):
    entities: Dict[str, list[Entity]]

class ExtractEntitiesBatchResponse(ExtractEntitiesResponse):
    progress: BatchProgress
