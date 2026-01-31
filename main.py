import os
import json
import time
from typing import List, Literal, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================
# Environment & OpenAI Setup
# =====================================================

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_RETRIES = 3

PRICING = {
    "gpt-4.1": (1.00, 3.00),  # input, output per 1M tokens
}

# =====================================================
# Shared Helpers
# =====================================================

def _tight_json_extract(text: str) -> str:
    """Best-effort extraction of a single JSON object from model output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        raise RuntimeError(f"No JSON object found in model output: {text!r}")
    return text[start:end]


def call_openai_json(
    system: str,
    user: str,
    *,
    max_retries: int = MAX_RETRIES,
    temperature: float = TEMPERATURE
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls the Responses API and expects a JSON object back.
    Returns: (parsed_json, meta)
    meta includes elapsed, usage, cost, raw_text.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()

            # NOTE:
            # We keep the "hard JSON guard" for robustness.
            # If you want stricter enforcement, you can try response_format
            # depending on your SDK/version support.
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )

            elapsed = time.perf_counter() - start
            usage = response.usage

            raw_text = (response.output_text or "").strip()
            if not raw_text:
                raise RuntimeError("Empty model output")

            json_text = _tight_json_extract(raw_text)
            parsed = json.loads(json_text)

            in_price, out_price = PRICING[MODEL]
            cost = (
                usage.input_tokens * in_price
                + usage.output_tokens * out_price
            ) / 1_000_000

            meta = {
                "elapsed": elapsed,
                "usage": usage,
                "cost": cost,
                "raw_text": raw_text,
            }
            return parsed, meta

        except Exception as e:
            last_error = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"OpenAI call failed: {last_error}")


# =====================================================
# Data Models (Structured, Arbiter-friendly)
# =====================================================

VerdictLabel = Literal["faithful", "mutated", "ambiguous"]

class GapAnalysis(BaseModel):
    missing_context_questions: List[str] = Field(
        default_factory=list,
        description="Questions that must be answered to judge faithfulness without guessing."
    )
    key_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimensions to check (scope, time, definitions, units, entity identity, qualifiers, etc.)."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)

class BuiltContext(BaseModel):
    # Think of this as your dynamic “committee context”
    distilled_reference: str = Field(..., description="Short, neutral distillation of the reference truth.")
    definitions: List[str] = Field(default_factory=list)
    scope_rules: List[str] = Field(default_factory=list)
    qualifier_rules: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    common_fallacies_to_avoid: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class RelevancyReport(BaseModel):
    relevant: bool
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class FactCheckReport(BaseModel):
    verdict: VerdictLabel
    reasoning: str

    # “Critical Semantic Audit” style flags:
    contradicts_reference: bool
    adds_new_unsupported_info: bool
    drops_required_qualifiers: bool
    shifts_scope_or_timeframe: bool
    changes_implication_or_intent: bool
    ambiguous_due_to_missing_context: bool

    confidence: float = Field(..., ge=0.0, le=1.0)
    key_mismatches: List[str] = Field(default_factory=list)
    preserved_points: List[str] = Field(default_factory=list)

class IntegrityReport(BaseModel):
    passes: bool
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class AgentVerdict(BaseModel):
    agent_name: str
    output: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)

class JuryResult(BaseModel):
    final_verdict: Literal["FAITHFUL", "MUTATED", "AMBIGUOUS"]
    summary_reason: str
    agent_outputs: List[AgentVerdict]
    score: Dict[str, float]


# =====================================================
# Agents
# =====================================================

class BaseAgent:
    name: str

    def run(self, *args, **kwargs):
        raise NotImplementedError


class ContextGapDetectorAgent(BaseAgent):
    name = "ContextGapDetector"

    def run(self, reference_truth: str, system_claim: str) -> GapAnalysis:
        system = (
            "You are a specialised context gap detector for truthfulness evaluation.\n"
            "Your job: identify what context is REQUIRED to judge whether the claim faithfully preserves the meaning of the reference.\n"
            "You must be strict: if the reference includes qualifiers, scope, timeframe, definitions, or implicit constraints, surface them.\n"
            "Return ONLY valid JSON."
        )

        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

TASK:
1) List the minimum missing-context questions needed to judge faithfulness WITHOUT guessing.
2) List key dimensions to audit (scope, timeframe, entity identity, units, definitions, qualifiers, causality, normative language, etc.).
3) Provide a confidence score for your gap analysis.

OUTPUT JSON SCHEMA:
{{
  "missing_context_questions": ["..."],
  "key_dimensions": ["..."],
  "confidence": 0.0
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return GapAnalysis(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid GapAnalysis output: {parsed}") from e


class ContextBuilderAgent(BaseAgent):
    name = "ContextBuilder"

    def run(self, reference_truth: str, gap: GapAnalysis) -> BuiltContext:
        system = (
            "You are a specialised context builder for semantic faithfulness checking.\n"
            "You MUST build context strictly from the REFERENCE TRUTH provided (no outside facts).\n"
            "Convert the reference into:\n"
            "- a neutral distilled summary,\n"
            "- definitions (if terms are used in a specific way),\n"
            "- scope rules (what is in/out of scope),\n"
            "- qualifier rules (must-keep constraints),\n"
            "- negative examples of common mistakes,\n"
            "- common fallacies to avoid when comparing claims.\n"
            "Return ONLY valid JSON."
        )

        questions = "\n".join(f"- {q}" for q in gap.missing_context_questions) or "- (none)"
        dims = "\n".join(f"- {d}" for d in gap.key_dimensions) or "- (none)"

        user = f"""
REFERENCE TRUTH:
{reference_truth}

GAP QUESTIONS TO ADDRESS:
{questions}

KEY AUDIT DIMENSIONS:
{dims}

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
  "confidence": 0.0
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return BuiltContext(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid BuiltContext output: {parsed}") from e


class RelevancyAssessorAgent(BaseAgent):
    name = "RelevancyAssessor"

    def run(self, context: BuiltContext, system_claim: str) -> RelevancyReport:
        system = (
            "You are a relevancy gate.\n"
            "Decide if the SYSTEM CLAIM is about the same topic/entity/event described by the distilled reference.\n"
            "Be conservative: if it is clearly unrelated, mark relevant=false.\n"
            "Return ONLY valid JSON."
        )

        user = f"""
DISTILLED REFERENCE:
{context.distilled_reference}

SYSTEM CLAIM:
{system_claim}

OUTPUT JSON SCHEMA:
{{
  "relevant": true,
  "reason": "string",
  "confidence": 0.0
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return RelevancyReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid RelevancyReport output: {parsed}") from e


class FactCheckerAgent(BaseAgent):
    name = "FactChecker"

    def run(self, reference_truth: str, system_claim: str, context: BuiltContext) -> FactCheckReport:
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

        # Inject dynamic rules like the paper's context-conditioned FactChecker prompt style.
        def _bullets(xs: List[str]) -> str:
            return "\n".join(f"- {x}" for x in xs) if xs else "- (none)"

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
  "preserved_points": ["..."]
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return FactCheckReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid FactCheckReport output: {parsed}") from e


class IntegrityValidatorAgent(BaseAgent):
    name = "IntegrityValidator"

    def run(self, factcheck: FactCheckReport) -> IntegrityReport:
        system = (
            "You are an integrity validator.\n"
            "Your job: sanity check the FactCheckReport for internal consistency.\n"
            "Flag issues such as: verdict says faithful but flags indicate contradiction; confidence mismatched with uncertainty; etc.\n"
            "Return ONLY valid JSON."
        )

        user = f"""
FACTCHECK REPORT:
{factcheck.model_dump_json(indent=2)}

OUTPUT JSON SCHEMA:
{{
  "passes": true,
  "issues": ["..."],
  "confidence": 0.0
}}
"""
        parsed, _ = call_openai_json(system, user)
        try:
            return IntegrityReport(**parsed)
        except ValidationError as e:
            raise RuntimeError(f"Invalid IntegrityReport output: {parsed}") from e


# =====================================================
# Deterministic Arbiter (Non-LLM)
# =====================================================

def arbitrate(
    relevancy: RelevancyReport,
    factcheck: FactCheckReport,
    integrity: IntegrityReport,
    context: BuiltContext,
    gap: GapAnalysis
) -> Tuple[str, str, Dict[str, float]]:
    """
    Deterministic decision logic inspired by the paper’s rule-based arbiter:
    - Reject (mutated) if contradiction/qualifier loss/scope shift/etc. are confidently present.
    - Faithful only if no critical flags and relevancy is true and integrity passes.
    - Otherwise ambiguous.
    """

    # Start with confidence-weighted scoring
    score = {"faithful": 0.0, "mutated": 0.0, "ambiguous": 0.0}

    # Hard gates
    if not relevancy.relevant and relevancy.confidence >= 0.70:
        score["mutated"] += 1.0
        return (
            "MUTATED",
            "Claim appears topically/entity-wise irrelevant to the reference.",
            score
        )

    if not integrity.passes and integrity.confidence >= 0.70:
        score["ambiguous"] += 1.0
        return (
            "AMBIGUOUS",
            "Integrity validator found internal inconsistencies in the fact-check report.",
            score
        )

    # Flag severity
    critical_flags = [
        ("contradicts_reference", 1.2),
        ("adds_new_unsupported_info", 0.9),
        ("drops_required_qualifiers", 1.1),
        ("shifts_scope_or_timeframe", 1.0),
        ("changes_implication_or_intent", 1.1),
    ]

    mutated_weight = 0.0
    for flag, w in critical_flags:
        if getattr(factcheck, flag):
            mutated_weight += w

    if factcheck.ambiguous_due_to_missing_context:
        score["ambiguous"] += 0.8 * factcheck.confidence

    # Apply factcheck verdict as a prior
    score[factcheck.verdict] += 1.0 * factcheck.confidence

    # Apply critical flags
    score["mutated"] += mutated_weight * factcheck.confidence

    # Mild bonus for context confidence (better context reduces ambiguity)
    score["faithful"] += 0.15 * context.confidence
    score["ambiguous"] += 0.10 * (1.0 - context.confidence) + 0.05 * (1.0 - gap.confidence)

    # Decide
    best = max(score, key=score.get)

    # Require margin to avoid flip-flopping
    sorted_scores = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_val = sorted_scores[0]
    second_val = sorted_scores[1][1]

    if top_val < second_val + 0.35:
        return (
            "AMBIGUOUS",
            "Scores are close; decision is interpretation-dependent given remaining uncertainty.",
            score
        )

    if top_label == "mutated":
        return ("MUTATED", "Deterministic arbiter found high-confidence semantic distortion flags.", score)
    if top_label == "faithful":
        return ("FAITHFUL", "Deterministic arbiter found meaning preserved with no high-severity drift.", score)
    return ("AMBIGUOUS", "Arbiter could not reach a stable margin.", score)


# =====================================================
# Orchestrator
# =====================================================

def run_committee(reference_truth: str, system_claim: str) -> JuryResult:
    gap_agent = ContextGapDetectorAgent()
    ctx_agent = ContextBuilderAgent()
    rel_agent = RelevancyAssessorAgent()
    fc_agent = FactCheckerAgent()
    iv_agent = IntegrityValidatorAgent()

    agent_outputs: List[AgentVerdict] = []

    gap = gap_agent.run(reference_truth, system_claim)
    agent_outputs.append(AgentVerdict(agent_name=gap_agent.name, output=gap.model_dump(), confidence=gap.confidence))

    context = ctx_agent.run(reference_truth, gap)
    agent_outputs.append(AgentVerdict(agent_name=ctx_agent.name, output=context.model_dump(), confidence=context.confidence))

    relevancy = rel_agent.run(context, system_claim)
    agent_outputs.append(AgentVerdict(agent_name=rel_agent.name, output=relevancy.model_dump(), confidence=relevancy.confidence))

    factcheck = fc_agent.run(reference_truth, system_claim, context)
    agent_outputs.append(AgentVerdict(agent_name=fc_agent.name, output=factcheck.model_dump(), confidence=factcheck.confidence))

    integrity = iv_agent.run(factcheck)
    agent_outputs.append(AgentVerdict(agent_name=iv_agent.name, output=integrity.model_dump(), confidence=integrity.confidence))

    final_verdict, summary_reason, score = arbitrate(relevancy, factcheck, integrity, context, gap)

    return JuryResult(
        final_verdict=final_verdict,
        summary_reason=summary_reason,
        agent_outputs=agent_outputs,
        score=score,
    )


# =====================================================
# Main
# =====================================================

def main():
    df = pd.read_csv("North_Star.csv")

    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    rows_to_check = [5, 6, 7, 12, 19]
    SWAP_FACT_AND_CLAIM = False

    rows = df.iloc[[i for i in rows_to_check if i in df.index]]

    print("\n====================================")
    print(" FACTTRACE – AI COMMITTEE-STYLE JURY")
    print("====================================\n")

    for idx, row in rows.iterrows():
        truth = str(row[TRUTH_COL])
        claim = str(row[CLAIM_COL])

        reference_truth, system_claim = (
            (claim, truth) if SWAP_FACT_AND_CLAIM else (truth, claim)
        )

        print("\n------------------------------------")
        print(f"ROW {idx}")
        print("------------------------------------")
        print("\nREFERENCE TRUTH:")
        print(reference_truth)
        print("\nSYSTEM CLAIM:")
        print(system_claim)

        result = run_committee(reference_truth, system_claim)

        print("\n--- AGENT OUTPUTS ---")
        for a in result.agent_outputs:
            print(f"\n[{a.agent_name}] (confidence={a.confidence:.2f})")
            print(json.dumps(a.output, indent=2, ensure_ascii=False))

        print("\n=== FINAL VERDICT ===")
        print(result.final_verdict)
        print("WHY:", result.summary_reason)
        print("SCORE:", {k: round(v, 3) for k, v in result.score.items()})
        print("\n")


if __name__ == "__main__":
    main()
