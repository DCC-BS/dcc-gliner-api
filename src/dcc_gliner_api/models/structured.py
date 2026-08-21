"""Request models for structured (JSON) data extraction endpoints."""

from pydantic import Field

from dcc_gliner_api.models.common import BatchOptions, ExtractOptions


class ExtractJsonRequest(ExtractOptions):
    text: str
    structures: dict[str, list[str]] = Field(
        ...,
        description='Record definitions, e.g. {"product": ["name::str::Full product name", "price"]}',
    )


class BatchExtractJsonRequest(ExtractOptions, BatchOptions):
    texts: list[str]
    structures: dict[str, list[str]]
