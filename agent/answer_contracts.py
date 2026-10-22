"""Contracts for grounded answers and the small-model completion gate."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .request_ledger import RequirementSpec


class SourceField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    information_id: str
    field: str = Field(min_length=1, max_length=100, description="facts.fields 내부의 상대 경로. 예: used_mib, entries.0.rss_kib")

    @field_validator("field")
    @classmethod
    def relative_field(cls, value):
        # Accept the container-qualified spelling before any basis/unit checks.
        # All remaining segments still pass the numeric source resolver.
        return value.removeprefix("fields.")


class Calculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=150)
    requirement_ids: list[str] = Field(min_length=1, max_length=16)
    operation: Literal["sum", "difference", "product", "ratio", "percent"]
    operands: list[SourceField] = Field(min_length=1, max_length=8)
    output_unit: Literal["native", "B", "KiB", "MiB", "GiB", "%", "percentage_points", "ratio"] = "native"


class AnswerStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    requirement_ids: list[str] = Field(min_length=1, max_length=16)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)
    # 12 model-selected quotes plus at most 12 command-derived support sources.
    quotes: list[str] = Field(default_factory=list, max_length=24)
    quote_ids: list[str] = Field(default_factory=list, max_length=12, exclude=True)


class RequirementCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    status: Literal["satisfied", "unmet", "unavailable"]
    reason: str = Field(min_length=1, max_length=300)
    rejection_matches_request: bool | None = Field(default=None, exclude=True)


class AnswerBlock(BaseModel):
    """Internal provenance for text that survives final rendering."""
    model_config = ConfigDict(extra="forbid")
    block_id: str
    text: str
    section: str
    requirement_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
