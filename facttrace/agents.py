"""LLM-backed agents used by the committee, plus a critic for reruns."""

from typing import List, Optional
from pydantic import ValidationError

from .openai_utils import call_openai_json
from .schemas import (
    BuiltContext,
    CriticReport,
    FactCheckReport,
    GapAnalysis,
    IntegrityReport,
    RelevancyReport,
)


class BaseAgent:
    name: str

    def run(self, *args, **kwargs):
        raise NotImplementedError


class ContextGapDetectorAgent(BaseAgent):
    name = "ContextGapDetector"

    def run(self, reference_truth: str, system_claim: str, critic_feedback: Optional[str] = None) -> GapAnalysis:
        system = (
            "You are a specialised context gap detector for truthfulness evaluation.\n"
            "Your job: identify what context is REQUIRED to judge whether the claim faithfully preserves the meaning of the reference.\n"
            "You must be strict: if the reference includes qualifiers, scope, timeframe, definitions, or implicit constraints, surface them.\n"
            "Return ONLY valid JSON."
        )

        feedback = critic_feedback or "(none)"

        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

CRITIC FEEDBACK (address explicitly):
{feedback}

TASK:
1) List the minimum missing-context questions needed to judge faithfulness WITHOUT guessing.
2) List key dimensions to audit (scope, timeframe, entity identity, units, definitions, qualifiers, causality, normative language, etc.).
3) Provide a confidence score for your gap analysis.
4) Provide a brief explanation (one or two sentences) of your reasoning and optionally warnings.

OUTPUT JSON SCHEMA:
{{
  "missing_context_questions": ["..."],
  "key_dimensions": ["..."],
  "confidence": 0.0,
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return GapAnalysis(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid GapAnalysis output: {parsed}") from e


class ContextBuilderAgent(BaseAgent):
    name = "ContextBuilder"

    def run(self, reference_truth: str, gap: GapAnalysis, critic_feedback: Optional[str] = None) -> BuiltContext:
        system = (
            "You are a specialised context builder for semantic faithfulness checking.\n"
            "You MUST build context strictly from the REFERENCE TRUTH provided (no outside facts).\n"
            "Return ONLY valid JSON."
        )

        questions = "\n".join(f"- {q}" for q in gap.missing_context_questions) or "- (none)"
        dims = "\n".join(f"- {d}" for d in gap.key_dimensions) or "- (none)"
        feedback = critic_feedback or "(none)"

        user = f"""
REFERENCE TRUTH:
{reference_truth}

GAP QUESTIONS TO ADDRESS:
{questions}

KEY AUDIT DIMENSIONS:
{dims}

CRITIC FEEDBACK (address explicitly):
{feedback}

CONSTRAINTS:
- Use ONLY information present in the reference truth.
- If something cannot be answered from the reference, explicitly note it as an unknown in scope_rules or qualifier_rules.

OUTPUT JSON SCHEMA:
{{
  "distilled_reference": "string",
  "definitions": ["..."],
  "scope_rules": ["..."],
  "qualifier_rules": ["..."],
  "negative_examples": ["..."],
  "common_fallacies_to_avoid": ["..."],
  "confidence": 0.0,
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return BuiltContext(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid BuiltContext output: {parsed}") from e


class RelevancyAssessorAgent(BaseAgent):
    name = "RelevancyAssessor"

    def run(self, context: BuiltContext, system_claim: str, critic_feedback: Optional[str] = None) -> RelevancyReport:
        system = (
            "You are a relevancy gate.\n"
            "Decide if the SYSTEM CLAIM is about the same topic/entity/event described by the distilled reference.\n"
            "Be conservative: if it is clearly unrelated, mark relevant=false.\n"
            "Return ONLY valid JSON."
        )

        feedback = critic_feedback or "(none)"

        user = f"""
DISTILLED REFERENCE:
{context.distilled_reference}

SYSTEM CLAIM:
{system_claim}

CRITIC FEEDBACK (address explicitly):
{feedback}

OUTPUT JSON SCHEMA:
{{
  "relevant": true,
  "reason": "string",
  "confidence": 0.0,
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return RelevancyReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid RelevancyReport output: {parsed}") from e


class FactCheckerAgent(BaseAgent):
    name = "FactChecker"

    def run(self, reference_truth: str, system_claim: str, context: BuiltContext, critic_feedback: Optional[str] = None) -> FactCheckReport:
        system = (
            "You are the Fact Checker.\n"
            "Goal: decide whether the SYSTEM CLAIM is faithful to the REFERENCE TRUTH.\n"
            "IMPORTANT: Literal overlap is NOT enough. Any semantic drift, qualifier loss, scope/timeframe shift, implication change, or added unsupported info => mutated.\n"
            "You must perform a Critical Semantic Audit:\n"
            "- Entity identity & granularity\n"
            "- Timeframe/scope\n"
            "- Qualifiers/hedges/modality (may/might/likely/only/except/at least)\n"
            "- Causality vs correlation\n"
            "- Normative language and implication shifts\n"
            "Return ONLY valid JSON."
        )

        def _bullets(xs: List[str]) -> str:
            return "\n".join(f"- {x}" for x in xs) if xs else "- (none)"

        feedback = critic_feedback or "(none)"

        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

DYNAMIC CONTEXT (MUST USE):
DEFINITIONS:
{_bullets(context.definitions)}

SCOPE RULES:
{_bullets(context.scope_rules)}

QUALIFIER RULES:
{_bullets(context.qualifier_rules)}

NEGATIVE EXAMPLES (what to avoid):
{_bullets(context.negative_examples)}

COMMON FALLACIES TO AVOID:
{_bullets(context.common_fallacies_to_avoid)}

CRITIC FEEDBACK (address explicitly):
{feedback}

OUTPUT JSON SCHEMA:
{{
  "verdict": "faithful|mutated|ambiguous",
  "reasoning": "string",
  "contradicts_reference": true,
  "adds_new_unsupported_info": true,
  "drops_required_qualifiers": true,
  "shifts_scope_or_timeframe": true,
  "changes_implication_or_intent": true,
  "ambiguous_due_to_missing_context": true,
  "confidence": 0.0,
  "key_mismatches": ["..."],
  "preserved_points": ["..."],
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return FactCheckReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid FactCheckReport output: {parsed}") from e


class IntegrityValidatorAgent(BaseAgent):
    name = "IntegrityValidator"

    def run(self, factcheck: FactCheckReport, critic_feedback: Optional[str] = None) -> IntegrityReport:
        system = (
            "You are an integrity validator.\n"
            "Your job: sanity check the FactCheckReport for internal consistency.\n"
            "Flag issues such as: verdict says faithful but flags indicate contradiction; confidence mismatched with uncertainty; etc.\n"
            "Return ONLY valid JSON."
        )

        feedback = critic_feedback or "(none)"

        user = f"""
FACTCHECK REPORT:
{factcheck.model_dump_json(indent=2)}

CRITIC FEEDBACK (address explicitly):
{feedback}

OUTPUT JSON SCHEMA:
{{
  "passes": true,
  "issues": ["..."],
  "confidence": 0.0,
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return IntegrityReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid IntegrityReport output: {parsed}") from e


class CriticAgent(BaseAgent):
    name = "Critic"

    def run(
        self,
        reference_truth: str,
        system_claim: str,
        gap: GapAnalysis,
        context: BuiltContext,
        relevancy: RelevancyReport,
        factcheck: FactCheckReport,
        integrity: IntegrityReport,
    ) -> CriticReport:
        system = (
            "You are a meta-critic supervising a committee of agents performing semantic faithfulness checking.\n"
            "Given all agent outputs, identify issues, inconsistencies, or missing analyses.\n"
            "When you find issues, propose reruns for specific agents with concrete instructions.\n"
            "Return ONLY valid JSON."
        )

        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

AGENT OUTPUTS (verbatim JSON):
- GAP: {gap.model_dump_json()}
- CONTEXT: {context.model_dump_json()}
- RELEVANCY: {relevancy.model_dump_json()}
- FACTCHECK: {factcheck.model_dump_json()}
- INTEGRITY: {integrity.model_dump_json()}

WHAT TO DO:
- List concrete issues (if any).
- For each rerun request, name the target_agent and give a precise instruction; keep the list short and actionable.
- Keep confidence calibrated.
- Provide a concise explanation and optional warnings.

OUTPUT JSON SCHEMA:
{{
  "issues": ["..."],
  "requests": [
    {{
      "target_agent": "ContextGapDetector|ContextBuilder|RelevancyAssessor|FactChecker|IntegrityValidator",
      "instruction": "string",
      "priority": "low|medium|high"
    }}
  ],
  "confidence": 0.0,
  "explanation": "string",
  "warnings": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return CriticReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid CriticReport output: {parsed}") from e


__all__ = [
    "BaseAgent",
    "ContextGapDetectorAgent",
    "ContextBuilderAgent",
    "RelevancyAssessorAgent",
    "FactCheckerAgent",
    "IntegrityValidatorAgent",
    "CriticAgent",
]
