"""Request models for multi-task schema extraction endpoints."""

from typing import Any

from pydantic import Field

from dcc_gliner_api.models.common import BatchOptions, ExtractOptions

SCHEMA_EXAMPLE = {
    "entities": {"person": "Names of people", "company": "Organization names"},
    "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"]}],
    "relations": ["works_for", "located_in"],
    "structures": {
        "product_info": {
            "fields": [
                {"name": "name", "dtype": "str", "description": "Product name"},
                {
                    "name": "price",
                    "validators": [{"pattern": r"^\$[\d,.]+$", "mode": "full"}],
                },
            ]
        }
    },
}


class ExtractRequest(ExtractOptions):
    text: str
    schema: dict[str, Any] = Field(
        ...,
        description="Multi-task schema: entities, classifications, relations, structures "
        "(fields support dtype, choices, description, threshold, validators)",
        examples=[SCHEMA_EXAMPLE],
    )


class BatchExtractRequest(ExtractOptions, BatchOptions):
    texts: list[str]
    schemas: dict[str, Any] | list[dict[str, Any]] = Field(
        ..., description="One schema for all texts, or one schema per text"
    )
