"""Shared option models and type aliases used across all request models."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

EntityTypes = Union[List[str], Dict[str, str]]
RelationTypes = Union[List[str], Dict[str, str]]
ValidatorSpec = Dict[str, Any]
FieldSpec = Union[str, Dict[str, Any]]
TasksSpec = Dict[str, Any]


class ExtractOptions(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    include_confidence: bool = Field(
        False, description="Attach confidence scores to results"
    )
    include_spans: bool = Field(
        False, description="Attach character spans (start/end) to results"
    )
    max_len: Optional[int] = Field(
        None, description="Max input tokens for the encoder window"
    )


class BatchOptions(BaseModel):
    batch_size: int = Field(8, ge=1, description="Batch size for batched inference")
