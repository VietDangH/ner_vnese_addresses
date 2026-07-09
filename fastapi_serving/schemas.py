# -*- coding: utf-8 -*-
"""Pydantic models for request validation and response formatting.

Keeping the wire format in one place makes the API self-documenting (these power
the Swagger UI at ``/docs``) and decoupled from the model internals.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="A single address string to tag.",
        examples=["123 đường lê lợi, phường bến nghé, quận 1, tp hcm"],
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("text must not be empty")
        return v


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="A list of address strings to tag.",
        examples=[["123 lê lợi, quận 1", "204/9 hải phòng, huyện cát hải"]],
    )

    @field_validator("texts")
    @classmethod
    def _has_content(cls, v: List[str]) -> List[str]:
        if not any(t and t.strip() for t in v):
            raise ValueError("texts must contain at least one non-empty string")
        return v


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class Entity(BaseModel):
    text: str = Field(..., description="Entity surface text from the input.")
    label: str = Field(..., description="Entity type, e.g. STREET, WARD, CITY.")
    start: int = Field(..., description="Start character offset in the input (-1 if unaligned).")
    end: int = Field(..., description="End character offset in the input (exclusive).")
    confidence: Optional[float] = Field(
        None, description="Model confidence for the span (mean over its tokens).")


class PredictResponse(BaseModel):
    text: str
    entities: List[Entity]
    latency_ms: Optional[float] = None


class BatchPredictResponse(BaseModel):
    count: int
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: Optional[str] = None
    labels: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Shape of error payloads (for the Swagger docs)."""
    detail: str
