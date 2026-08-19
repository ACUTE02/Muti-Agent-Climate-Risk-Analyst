"""Request and response shapes for the local API.

The response deliberately **promotes this project's honesty mechanisms to
structured fields** rather than leaving them buried in report prose. A client
should never have to parse English to find out whether a figure was verified, or
whether a forecast exists at all.

Three rules this schema exists to enforce:

1. Grounding status is an explicit field, not something inferred from the text.
2. Per-horizon confidence labels stay distinct — `validated`, `weak/directional`
   and `no skill ...` are never collapsed server-side into one confidence number.
   Collapsing them would undo five phases of honesty in a single line of code.
3. Missing data is stated, not omitted. `forecast_available: false` and "no
   sourced yield-impact estimate available" travel as explicit flags with
   reasons attached, never as an absent key or a bare null.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    request: str = Field(..., min_length=1,
                         description="The natural-language question to answer.")
    region: str | None = Field(None, description="Force a region, e.g. 'barmer'.")
    risk_types: list[str] | None = Field(
        None, description="Force risk types, e.g. ['drought', 'heat_stress'].")
    month: str | None = Field(None, description="Target month, e.g. '2024-05'.")
    crop: str | None = Field(None, description="Force a crop, e.g. 'wheat'.")


class GroundingStatus(BaseModel):
    """The mechanical checker's verdict, surfaced as structure."""

    status: str = Field(..., description="'clean', 'warning' or 'not_generated'.")
    grounded: bool
    total_checked: int
    unverified_numbers: list[str] = []
    report_missing: bool = False
    explanation: str


class HorizonConfidence(BaseModel):
    """One forecast horizon's measured reliability, kept distinct on purpose."""

    horizon: str
    skill_score: float
    method: str
    label: str
    reliable: bool = Field(
        ..., description="True only for a 'validated' label — never a rounding "
                         "of the skill score into a traffic light.")


class MissingDataFlag(BaseModel):
    """Something this system explicitly does not have, and why."""

    what: str
    available: bool = False
    reason: str


class ReportResponse(BaseModel):
    request: str
    report: str
    grounding: GroundingStatus
    horizon_confidence: list[HorizonConfidence] = []
    missing_data: list[MissingDataFlag] = []
    tools_called: list[str] = []
    tool_outputs: dict[str, Any] = {}
    retrieved_sources: list[dict[str, Any]] = []
    external_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Live third-party sources (IMD's peers: NASA POWER, "
                    "data.gov.in), each with its publisher and citation. Kept "
                    "separate from tool_outputs on purpose: these are other "
                    "organisations' figures, never this project's measurements. "
                    "Unavailable sources appear here too, with a reason.")
    warnings: list[str] = []
    quota: dict[str, Any] = {}


class QuotaStatus(BaseModel):
    daily_call_budget: int
    calls_used_today: int
    calls_remaining_today: int
    typical_calls_per_report: int
    worst_case_calls_per_report: int
    note: str


class HealthResponse(BaseModel):
    status: str
    data_currency: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-region input currency: how recent the data behind a live "
                    "forecast actually is, and which input is the binding limit.")
    chroma_index_ready: bool
    chroma_chunks: int | None = None
    forecast_artifacts_ready: bool
    missing_artifacts: list[str] = []
    api_key_present: bool
    quota: QuotaStatus
    note: str
