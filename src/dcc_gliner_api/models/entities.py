"""Request models for entity extraction endpoints."""
from typing import List

from pydantic import BaseModel, Field

from .common import BatchOptions, EntityTypes, ExtractOptions


class PredictModel(BaseModel):
    """Legacy convenience endpoint payload (fixed German label set)."""
    text: str


class ExtractEntitiesRequest(ExtractOptions):
    text: str
    entity_types: EntityTypes = Field(..., description="List of labels, or dict label -> description")


class BatchExtractEntitiesRequest(ExtractOptions, BatchOptions):
    texts: List[str]
    entity_types: EntityTypes
