# facttrace_onefile.py
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

MODEL = os.getenv("FACTTRACE_MODEL", "gpt-4.1")
TEMPERATURE = 0.0
MAX_RETRIES = 3


# =====================================================
# JSON Helpers
# =====================================================

def _tight_json_extract(text: str) -> str:
    """
    Best-effort extraction of a single JSON object from model output.
    Works even if the model adds prose around it.
    """
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
    temperature: float = TEMPERATURE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls the Responses API and expects a JSON object back.
    Returns (parsed_json, meta). Meta is kept for debugging but never printed.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()

            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )

            elapsed = time.perf_counter() - start
            usage = getattr(response, "usage", None)
            raw_text = (getattr(response, "output_text", None) or "").strip()
            if not raw_text:
                raise RuntimeError("Empty model output")

            json_text = _tight_json_extract(raw_text)
            parsed = json.loads(json_text)

            meta = {
                "elapsed": elapsed,
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                },
                "raw_text": raw_text,
            }
            return parsed, meta

        except Exception as e:
            last_error = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"OpenAI call failed: {last_error}")


# =====================================================
# Data Models
# =====================================================

FinalVerdict = Literal["FAITHFUL", "MUTATED", "AMBIGUOUS"]

class SufficiencyReport(BaseModel):
    sufficient: bool
    reasoning: str
    needed_info: List[str] = Field(default_factory=list)
    key_dimensions: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

SpecialistName = Literal[
    "numbers",
    "entities",
    "time_qualifiers",
    "relevance",
    "causality",
    "definitions",
]

class SpecialistReport(BaseModel):
    specialist: SpecialistName
    passed: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class ArbiterReport(BaseModel):
    final_verdict: Literal["FAITHFUL", "MUTATED"]
    summary_reason: str
    decisive_points: List[str] = Field(default_factory=list)
    required_extra_info: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class ArbiterAuditReport(BaseModel):
    approved: bool
    audit_reason: str
    detected_flaws: List[str] = Field(default_factory=list)
    revised_final_verdict: Optional[FinalVerdict] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class FinalSummary(BaseModel):
    verdict: FinalVerdict
    reason: str
    key_points: List[str] = Field(default_factory=list)

class CommitteeResult(BaseModel):
    row_index: int
    reference_truth: str
    system_claim: str

    sufficiency: SufficiencyReport
    specialists: List[SpecialistReport] = Field(default_factory=list)

    arbiter_1: Optional[ArbiterReport] = None
    arbiter_2: Optional[ArbiterAuditReport] = None

    final_verdict: FinalVerdict
    final_reason: str

    final_summary: Optional[FinalSummary] = None

    meta: Dict[str, Any] = Field(default_factory=dict)


# =====================================================
# Agents
# =====================================================

def agent_context_sufficiency(reference_truth: str, system_claim: str) -> Tuple[SufficiencyReport, Dict[str, Any]]:
    system = (
        "You are a strict context sufficiency judge for semantic faithfulness.\n"
        "Decide whether REFERENCE TRUTH contains enough information to judge whether SYSTEM CLAIM\n"
        "faithfully preserves the meaning.\n"
        "Be strict about: entity identity, timeframe, scope, qualifiers, definitions, missing numbers/units.\n"
        "If insufficient, set sufficient=false and list the minimum extra info needed.\n"
        "Return ONLY valid JSON."
    )

    user = (
        "REFERENCE TRUTH:\n"
        f"{reference_truth}\n\n"
        "SYSTEM CLAIM:\n"
        f"{system_claim}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "sufficient": true,\n'
        '  "reasoning": "string",\n'
        '  "needed_info": ["..."],\n'
        '  "key_dimensions": ["..."],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

    parsed, meta = call_openai_json(system, user)
    try:
        return SufficiencyReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SufficiencyReport output: {parsed}") from e


def _agent_specialist_generic(
    specialist: SpecialistName,
    system_prompt: str,
    reference_truth: str,
    system_claim: str,
) -> Tuple[SpecialistReport, Dict[str, Any]]:
    system = system_prompt + "\nReturn ONLY valid JSON."

    user = (
        "REFERENCE TRUTH:\n"
        f"{reference_truth}\n\n"
        "SYSTEM CLAIM:\n"
        f"{system_claim}\n\n"
        "Return JSON:\n"
        "{\n"
        f'  "specialist": "{specialist}",\n'
        '  "passed": true,\n'
        '  "reasoning": "string",\n'
        '  "issues": ["..."],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

    parsed, meta = call_openai_json(system, user)
    try:
        return SpecialistReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SpecialistReport({specialist}) output: {parsed}") from e


SPECIALIST_PROMPTS: Dict[SpecialistName, str] = {
    "relevance": (
        "You are a specialist checker for RELEVANCE / TOPIC ALIGNMENT.\n"
        "Decide whether the SYSTEM CLAIM is about the same core subject as the REFERENCE TRUTH.\n"
        "Fail if there is topic drift, non-sequitur, or cherry-picking that changes the main point.\n"
        "If material irrelevance exists, passed=false and list issues."
    ),
    "numbers": (
        "You are a specialist checker for NUMBERS, UNITS, quantities, magnitudes, and arithmetic consistency.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for number/range/rounding/unit changes and contradictions.\n"
        "If any material mismatch, passed=false and list issues."
    ),
    "entities": (
        "You are a specialist checker for ENTITIES: people, places, organisations, products, identifiers.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for entity substitution, missing/added entities,\n"
        "relationship changes (who did what to whom), or ambiguity that changes meaning.\n"
        "If any material mismatch or unsupported addition, passed=false and list issues."
    ),
    "time_qualifiers": (
        "You are a specialist checker for TIMEFRAMES, QUALIFIERS, modality, and scope.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for timeframe drift, qualifier drift (may/must etc.),\n"
        "and scope drift (subset/superset, conditions removed).\n"
        "If any material mismatch or unsupported strengthening/weakening, passed=false and list issues."
    ),
    "causality": (
        "You are a specialist checker for CAUSALITY and INFERENCE.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for added/removed causal links (caused/led to/due to),\n"
        "or correlation upgraded to causation, or mechanisms asserted without support.\n"
        "If any unsupported causality is introduced (or removed when essential), passed=false and list issues."
    ),
    "definitions": (
        "You are a specialist checker for DEFINITIONS and TERM SUBSTITUTION.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for term replacements that change meaning,\n"
        "definition drift, category/type errors, and technical/legal loosening or strengthening.\n"
        "If any material definition/term drift is present, passed=false and list issues."
    ),
}


def agent_arbiter_1(
    reference_truth: str,
    system_claim: str,
    specialists: List[SpecialistReport],
) -> Tuple[ArbiterReport, Dict[str, Any]]:
    system = (
        "You are Arbiter #1 producing a final verdict on semantic faithfulness.\n"
        "Rules:\n"
        "- If ANY specialist passed=false, final_verdict MUST be MUTATED.\n"
        "- If ALL specialists passed=true, final_verdict MUST be FAITHFUL.\n"
        "Return ONLY valid JSON."
    )

    user = (
        "REFERENCE TRUTH:\n"
        f"{reference_truth}\n\n"
        "SYSTEM CLAIM:\n"
        f"{system_claim}\n\n"
        "SPECIALISTS:\n"
        f"{[s.model_dump() for s in specialists]}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "final_verdict": "FAITHFUL|MUTATED",\n'
        '  "summary_reason": "string",\n'
        '  "decisive_points": ["..."],\n'
        '  "required_extra_info": [],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

    parsed, meta = call_openai_json(system, user)
    try:
        return ArbiterReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid ArbiterReport output: {parsed}") from e


def agent_arbiter_2_audit(
    reference_truth: str,
    system_claim: str,
    specialists: List[SpecialistReport],
    arbiter_1: ArbiterReport,
) -> Tuple[ArbiterAuditReport, Dict[str, Any]]:
    system = (
        "You are Arbiter #2 auditing Arbiter #1 for logical soundness.\n"
        "Your job:\n"
        "- Check Arbiter #1 verdict matches specialist results (rule-following).\n"
        "- Check Arbiter #1 reasoning aligns with specialist issues and does not invent facts.\n"
        "Output:\n"
        "- approved=true if Arbiter #1 reasoning is sound.\n"
        "- approved=false if reasoning is debatable/unsound; then revised_final_verdict MUST be AMBIGUOUS.\n"
        "Return ONLY valid JSON."
    )

    user = (
        "REFERENCE TRUTH:\n"
        f"{reference_truth}\n\n"
        "SYSTEM CLAIM:\n"
        f"{system_claim}\n\n"
        "SPECIALISTS:\n"
        f"{[s.model_dump() for s in specialists]}\n\n"
        "ARBITER_1_OUTPUT:\n"
        f"{arbiter_1.model_dump()}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "approved": true,\n'
        '  "audit_reason": "string",\n'
        '  "detected_flaws": ["..."],\n'
        '  "revised_final_verdict": null,\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

    parsed, meta = call_openai_json(system, user)
    try:
        return ArbiterAuditReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid ArbiterAuditReport output: {parsed}") from e


def agent_final_summariser(res: CommitteeResult) -> Tuple[FinalSummary, Dict[str, Any]]:
    system = (
        "You are a concise summariser.\n"
        "Given the structured jury outcome, produce a very short verdict + reason.\n"
        "Do not add new facts. Keep reason one sentence. Provide up to 3 key points.\n"
        "Return ONLY valid JSON."
    )

    user = (
        "INPUT_RESULT_JSON:\n"
        f"{res.model_dump()}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "verdict": "FAITHFUL|MUTATED|AMBIGUOUS",\n'
        '  "reason": "string",\n'
        '  "key_points": ["..."]\n'
        "}\n"
    )

    parsed, meta = call_openai_json(system, user)
    try:
        return FinalSummary(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid FinalSummary output: {parsed}") from e


# =====================================================
# Orchestrator
# =====================================================

DEFAULT_SPECIALISTS: List[SpecialistName] = [
    "relevance",
    "numbers",
    "entities",
    "time_qualifiers",
    "causality",
    "definitions",
]


def run_committee(reference_truth: str, system_claim: str, row_index: int) -> CommitteeResult:
    meta: Dict[str, Any] = {}

    # Step 1) Context sufficiency gate
    suff, m_suff = agent_context_sufficiency(reference_truth, system_claim)
    meta["sufficiency"] = m_suff

    if not suff.sufficient:
        final_verdict: FinalVerdict = "AMBIGUOUS"
        needed = ", ".join(suff.needed_info) if suff.needed_info else "more supporting detail"
        final_reason = f"Insufficient context to judge faithfulness. Needed: {needed}"

        res = CommitteeResult(
            row_index=row_index,
            reference_truth=reference_truth,
            system_claim=system_claim,
            sufficiency=suff,
            specialists=[],
            arbiter_1=None,
            arbiter_2=None,
            final_verdict=final_verdict,
            final_reason=final_reason,
            meta=meta,
        )

        summary, m_sum = agent_final_summariser(res)
        res.final_summary = summary
        meta["final_summary"] = m_sum
        return res

    # Step 2) Specialists
    specialists: List[SpecialistReport] = []
    for name in DEFAULT_SPECIALISTS:
        rep, m = _agent_specialist_generic(name, SPECIALIST_PROMPTS[name], reference_truth, system_claim)
        specialists.append(rep)
        meta[f"spec_{name}"] = m

    # Step 3) Arbiter #1
    arb1, m_a1 = agent_arbiter_1(reference_truth, system_claim, specialists)
    meta["arbiter_1"] = m_a1

    # Step 4) Arbiter #2 audit
    arb2, m_a2 = agent_arbiter_2_audit(reference_truth, system_claim, specialists, arb1)
    meta["arbiter_2"] = m_a2

    # Step 5) Final verdict
    if not arb2.approved:
        final_verdict = "AMBIGUOUS"
        final_reason = f"Debatable: Arbiter #2 did not approve Arbiter #1 reasoning. {arb2.audit_reason}"
    else:
        final_verdict = arb1.final_verdict
        final_reason = arb1.summary_reason

    res = CommitteeResult(
        row_index=row_index,
        reference_truth=reference_truth,
        system_claim=system_claim,
        sufficiency=suff,
        specialists=specialists,
        arbiter_1=arb1,
        arbiter_2=arb2,
        final_verdict=final_verdict,
        final_reason=final_reason,
        meta=meta,
    )

    # Step 6) Final summariser
    summary, m_sum = agent_final_summariser(res)
    res.final_summary = summary
    meta["final_summary"] = m_sum

    return res


# =====================================================
# Output Helpers (human, step-by-step; no perf/diagnostics)
# =====================================================

def _clip(s: str, n: int = 900) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def print_result(res: CommitteeResult) -> None:
    print("\n" + "=" * 80)
    print(f"ROW {res.row_index}")
    print("=" * 80)

    print("\nStep 0 — Inputs")
    print(f"- Reference truth: {_clip(res.reference_truth, 350)}")
    print(f"- System claim:    {_clip(res.system_claim, 350)}")

    print("\nStep 1 — Context sufficiency")
    print(f"- Sufficient: {res.sufficiency.sufficient}")
    print(f"- Reasoning:  {_clip(res.sufficiency.reasoning, 700)}")
    if res.sufficiency.key_dimensions:
        print(f"- Key checks: {', '.join(res.sufficiency.key_dimensions[:8])}")
    print(f"- Confidence: {res.sufficiency.confidence:.2f}")

    if not res.sufficiency.sufficient:
        if res.sufficiency.needed_info:
            print("\nStep 1b — Needed info to decide")
            for x in res.sufficiency.needed_info[:12]:
                print(f"- {_clip(x, 260)}")

        print("\nStep 2 — Final decision")
        print(f"- VERDICT: {res.final_verdict}")
        print(f"- REASON:  {_clip(res.final_reason, 420)}")

        if res.final_summary:
            print("\nStep 3 — Concise summary")
            print(f"- Verdict: {res.final_summary.verdict}")
            print(f"- Reason:  {_clip(res.final_summary.reason, 240)}")
            for kp in res.final_summary.key_points[:3]:
                print(f"  • {_clip(kp, 240)}")
        return

    print("\nStep 2 — Specialist checks")
    for s in res.specialists:
        status = "PASS" if s.passed else "FAIL"
        print(f"- [{s.specialist}] {status} (conf={s.confidence:.2f})")
        print(f"  Reasoning: {_clip(s.reasoning, 450)}")
        for iss in s.issues[:4]:
            print(f"  • {_clip(iss, 260)}")

    print("\nStep 3 — Arbiter #1")
    if res.arbiter_1:
        print(f"- Verdict:   {res.arbiter_1.final_verdict}")
        print(f"- Reason:    {_clip(res.arbiter_1.summary_reason, 450)}")
        for p in res.arbiter_1.decisive_points[:5]:
            print(f"  • {_clip(p, 260)}")
        print(f"- Confidence: {res.arbiter_1.confidence:.2f}")

    print("\nStep 4 — Arbiter #2 audit")
    if res.arbiter_2:
        print(f"- Approved:  {res.arbiter_2.approved}")
        print(f"- Reason:    {_clip(res.arbiter_2.audit_reason, 450)}")
        if res.arbiter_2.detected_flaws:
            for f in res.arbiter_2.detected_flaws[:5]:
                print(f"  • {_clip(f, 260)}")
        print(f"- Confidence: {res.arbiter_2.confidence:.2f}")

    print("\nStep 5 — Final decision (end)")
    print(f"- VERDICT: {res.final_verdict}")
    print(f"- REASON:  {_clip(res.final_reason, 420)}")

    if res.final_summary:
        print("\nStep 6 — Concise summary")
        print(f"- Verdict: {res.final_summary.verdict}")
        print(f"- Reason:  {_clip(res.final_summary.reason, 240)}")
        for kp in res.final_summary.key_points[:3]:
            print(f"  • {_clip(kp, 240)}")


# =====================================================
# Main
# =====================================================

def main():
    df = pd.read_csv("North_Star.csv")

    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    rows_to_check = [5, 8]  # only these rows
    SWAP_FACT_AND_CLAIM = False  # set True if your CSV columns are reversed

    rows = df.iloc[[i for i in rows_to_check if i in df.index]]

    print("\nFACTTRACE — SUFFICIENCY-GATED SPECIALIST JURY + DOUBLE ARBITER + FINAL SUMMARY")
    print(f"Model: {MODEL}")
    print("-" * 72)

    for idx, row in rows.iterrows():
        truth = str(row[TRUTH_COL])
        claim = str(row[CLAIM_COL])

        reference_truth, system_claim = (claim, truth) if SWAP_FACT_AND_CLAIM else (truth, claim)

        res = run_committee(reference_truth, system_claim, row_index=int(idx))
        print_result(res)


if __name__ == "__main__":
    main()
