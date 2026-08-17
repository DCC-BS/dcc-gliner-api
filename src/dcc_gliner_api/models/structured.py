"""Request models for structured (JSON) data extraction endpoints."""

from typing import Dict, List

from pydantic import Field

from .common import BatchOptions, ExtractOptions


class ExtractJsonRequest(ExtractOptions):
    text: str
    structures: Dict[str, List[str]] = Field(
        ...,
        description='Record definitions, e.g. {"product": ["name::str::Full product name", "price"]}',
    )


class BatchExtractJsonRequest(ExtractOptions, BatchOptions):
    texts: List[str]
    structures: Dict[str, List[str]]
