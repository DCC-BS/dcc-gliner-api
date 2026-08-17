"""Request models for relation extraction endpoints."""

from typing import List

from pydantic import Field

from .common import BatchOptions, ExtractOptions, RelationTypes


class ExtractRelationsRequest(ExtractOptions):
    text: str
    relation_types: RelationTypes = Field(
        ..., description="List of relations, or dict relation -> description"
    )


class BatchExtractRelationsRequest(ExtractOptions, BatchOptions):
    texts: List[str]
    relation_types: RelationTypes
