"""Request models for multi-task schema extraction endpoints."""

from typing import Any, Dict, List, Union

from pydantic import Field

from .common import BatchOptions, ExtractOptions

SCHEMA_EXAMPLE = {
    "entities": {"person": "Names of people", "company": "Organization names"},
    "classifications": [
        {"task": "sentiment", "labels": ["positive", "negative", "neutral"]}
    ],
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
    schema: Dict[str, Any] = Field(
        ...,
        description="Multi-task schema: entities, classifications, relations, structures "
        "(fields support dtype, choices, description, threshold, validators)",
        examples=[SCHEMA_EXAMPLE],
    )


class BatchExtractRequest(ExtractOptions, BatchOptions):
    texts: List[str]
    schemas: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., description="One schema for all texts, or one schema per text"
    )
