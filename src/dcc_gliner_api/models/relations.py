"""Request models for relation extraction endpoints."""

from pydantic import Field

from dcc_gliner_api.models.common import BatchOptions, ExtractOptions, RelationTypes


class ExtractRelationsRequest(ExtractOptions):
    text: str
    relation_types: RelationTypes = Field(
        ..., description="List of relations, or dict relation -> description"
    )


class BatchExtractRelationsRequest(ExtractOptions, BatchOptions):
    texts: list[str]
    relation_types: RelationTypes
