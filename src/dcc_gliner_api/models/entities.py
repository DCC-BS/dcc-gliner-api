"""Request models for entity extraction endpoints."""

from pydantic import Field

from .common import BatchOptions, EntityTypes, ExtractOptions


class ExtractEntitiesRequest(ExtractOptions):
    text: str
    entity_types: EntityTypes = Field(
        ..., description="List of labels, or dict label -> description"
    )


class BatchExtractEntitiesRequest(ExtractOptions, BatchOptions):
    texts: list[str]
    entity_types: EntityTypes
