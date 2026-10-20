"""Contracts for grounded answers and the small-model completion gate."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .request_ledger import RequirementSpec


class SourceField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    information_id: str
    field: str = Field(min_length=1, max_length=100)


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
    quotes: list[str] = Field(min_length=1, max_length=12)


class RequirementCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    status: Literal["satisfied", "unmet", "unavailable"]
    reason: str = Field(min_length=1, max_length=300)
