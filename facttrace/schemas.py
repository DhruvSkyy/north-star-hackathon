"""Pydantic models and shared types."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

VerdictLabel = Literal["faithful", "mutated", "ambiguous"]


class GapAnalysis(BaseModel):
    missing_context_questions: List[str] = Field(
        default_factory=list,
        description="Questions needed to judge faithfulness without guessing.",
    )
    key_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimensions to check (scope, time, definitions, units, etc.).",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class BuiltContext(BaseModel):
    distilled_reference: str = Field(
        ..., description="Short, neutral distillation of the reference truth."
    )
    definitions: List[str] = Field(default_factory=list)
    scope_rules: List[str] = Field(default_factory=list)
    qualifier_rules: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    common_fallacies_to_avoid: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class RelevancyReport(BaseModel):
    relevant: bool
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    verdict: VerdictLabel
    reasoning: str
    contradicts_reference: bool
    adds_new_unsupported_info: bool
    drops_required_qualifiers: bool
    shifts_scope_or_timeframe: bool
    changes_implication_or_intent: bool
    ambiguous_due_to_missing_context: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_mismatches: List[str] = Field(default_factory=list)
    preserved_points: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class IntegrityReport(BaseModel):
    passes: bool
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class CriticRequest(BaseModel):
    target_agent: Literal[
        "ContextGapDetector",
        "ContextBuilder",
        "RelevancyAssessor",
        "FactChecker",
        "IntegrityValidator",
    ]
    instruction: str
    priority: Literal["low", "medium", "high"] = "medium"


class CriticReport(BaseModel):
    issues: List[str] = Field(default_factory=list)
    requests: List[CriticRequest] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class AgentVerdict(BaseModel):
    agent_name: str
    output: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)


class JuryResult(BaseModel):
    final_verdict: Literal["FAITHFUL", "MUTATED", "AMBIGUOUS"]
    summary_reason: str
    agent_outputs: List[AgentVerdict]
    score: Dict[str, float]
    critic: Optional[CriticReport] = None
    reruns: Dict[str, int] = Field(default_factory=dict)


__all__ = [
    "VerdictLabel",
    "GapAnalysis",
    "BuiltContext",
    "RelevancyReport",
    "FactCheckReport",
    "IntegrityReport",
    "CriticRequest",
    "CriticReport",
    "AgentVerdict",
    "JuryResult",
]
