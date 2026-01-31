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
    Returns (parsed_json, meta).
    meta includes elapsed, usage (if available), raw_text.
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

class SpecialistReport(BaseModel):
    specialist: Literal["numbers", "entities", "time_qualifiers"]
    passed: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class ArbiterReport(BaseModel):
    final_verdict: FinalVerdict
    summary_reason: str
    decisive_points: List[str] = Field(default_factory=list)
    required_extra_info: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class CommitteeResult(BaseModel):
    row_index: int
    reference_truth: str
    system_claim: str

    sufficiency: SufficiencyReport
    specialists: List[SpecialistReport]
    arbiter: ArbiterReport

    meta: Dict[str, Any] = Field(default_factory=dict)


# =====================================================
# Agents
# =====================================================

def agent_context_sufficiency(reference_truth: str, system_claim: str) -> Tuple[SufficiencyReport, Dict[str, Any]]:
    system = (
        "You are a strict context sufficiency judge for semantic faithfulness.\n"
        "You DO NOT have web access.\n"
        "Task: decide whether the provided REFERENCE TRUTH contains enough information to judge whether\n"
        "the SYSTEM CLAIM faithfully preserves its meaning.\n"
        "Be strict about: entity identity, timeframe, scope, qualifiers, definitions, missing numbers/units.\n"
        "If insufficient, set sufficient=false and list the minimum extra info needed to decide.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

Return JSON:
{{
  "sufficient": true,
  "reasoning": "string",
  "needed_info": ["..."],
  "key_dimensions": ["..."],
  "confidence": 0.0
}}
"""
    parsed, meta = call_openai_json(system, user)
    try:
        return SufficiencyReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SufficiencyReport output: {parsed}") from e


def agent_specialist_numbers(reference_truth: str, system_claim: str) -> Tuple[SpecialistReport, Dict[str, Any]]:
    system = (
        "You are a specialist checker for NUMBERS, UNITS, quantities, magnitudes, and arithmetic consistency.\n"
        "You DO NOT have web access.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for:\n"
        "- number changes (including ranges, approximations, rounding)\n"
        "- unit changes (%, £, km, years, etc.)\n"
        "- added/removed numeric qualifiers (at least, exactly, about)\n"
        "- arithmetic contradictions\n"
        "If there is any material mismatch, passed=false and list issues.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

Return JSON:
{{
  "specialist": "numbers",
  "passed": true,
  "reasoning": "string",
  "issues": ["..."],
  "confidence": 0.0
}}
"""
    parsed, meta = call_openai_json(system, user)
    try:
        return SpecialistReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SpecialistReport(numbers) output: {parsed}") from e


def agent_specialist_entities(reference_truth: str, system_claim: str) -> Tuple[SpecialistReport, Dict[str, Any]]:
    system = (
        "You are a specialist checker for ENTITIES: names of people, places, organisations, products, and identifiers.\n"
        "You DO NOT have web access.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for:\n"
        "- entity substitution (different person/place/org)\n"
        "- missing/added entities\n"
        "- ambiguous pronoun resolution causing drift\n"
        "- changed relationships (who did what to whom)\n"
        "If any material mismatch or unsupported addition, passed=false and list issues.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

Return JSON:
{{
  "specialist": "entities",
  "passed": true,
  "reasoning": "string",
  "issues": ["..."],
  "confidence": 0.0
}}
"""
    parsed, meta = call_openai_json(system, user)
    try:
        return SpecialistReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SpecialistReport(entities) output: {parsed}") from e


def agent_specialist_time_qualifiers(reference_truth: str, system_claim: str) -> Tuple[SpecialistReport, Dict[str, Any]]:
    system = (
        "You are a specialist checker for TIMEFRAMES, QUALIFIERS, modality, and scope.\n"
        "You DO NOT have web access.\n"
        "Compare SYSTEM CLAIM against REFERENCE TRUTH for:\n"
        "- timeframe drift (was vs is, dates, 'recently', 'in 2020', etc.)\n"
        "- qualifier drift (may/must, likely/definitely, some/most/all)\n"
        "- scope drift (subset vs superset, conditions removed)\n"
        "- causality claims added (X caused Y) when not present\n"
        "If any material mismatch or unsupported strengthening/weakening, passed=false and list issues.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

Return JSON:
{{
  "specialist": "time_qualifiers",
  "passed": true,
  "reasoning": "string",
  "issues": ["..."],
 F"  "confidence": 0.0
}}
"""
    # NOTE: There is a stray 'F"' risk in some editors; keep this string literal exact.
    # We'll build it safely to avoid accidental corruption.
    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

Return JSON:
{{
  "specialist": "time_qualifiers",
  "passed": true,
  "reasoning": "string",
  "issues": ["..."],
  "confidence": 0.0
}}
"""
    parsed, meta = call_openai_json(system, user)
    try:
        return SpecialistReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid SpecialistReport(time_qualifiers) output: {parsed}") from e


def agent_arbiter(
    reference_truth: str,
    system_claim: str,
    sufficiency: SufficiencyReport,
    specialists: List[SpecialistReport],
) -> Tuple[ArbiterReport, Dict[str, Any]]:
    system = (
        "You are an arbiter that outputs the final verdict on semantic faithfulness.\n"
        "You DO NOT have web access.\n"
        "Rules:\n"
        "- If sufficiency.sufficient is false, final_verdict MUST be AMBIGUOUS.\n"
        "- If sufficiency is true, but ANY specialist passed=false, final_verdict MUST be MUTATED.\n"
        "- If sufficiency is true and ALL specialists passed=true, final_verdict MUST be FAITHFUL.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

SUFFICIENCY:
sufficient: {sufficiency.sufficient}
reasoning: {sufficiency.reasoning}
needed_info: {sufficiency.needed_info}
key_dimensions: {sufficiency.key_dimensions}
confidence: {sufficiency.confidence}

SPECIALISTS:
{[s.model_dump() for s in specialists]}

Return JSON:
{{
  "final_verdict": "FAITHFUL|MUTATED|AMBIGUOUS",
  "summary_reason": "string",
  "decisive_points": ["..."],
  "required_extra_info": ["..."],
  "confidence": 0.0
}}
"""
    parsed, meta = call_openai_json(system, user)
    try:
        return ArbiterReport(**parsed), meta
    except ValidationError as e:
        raise RuntimeError(f"Invalid ArbiterReport output: {parsed}") from e


# =====================================================
# Orchestrator
# =====================================================

def run_committee(reference_truth: str, system_claim: str, row_index: int) -> CommitteeResult:
    meta: Dict[str, Any] = {}

    suff, m1 = agent_context_sufficiency(reference_truth, system_claim)
    meta["sufficiency"] = m1

    specialists: List[SpecialistReport] = []
    if suff.sufficient:
        nrep, mn = agent_specialist_numbers(reference_truth, system_claim)
        erep, me = agent_specialist_entities(reference_truth, system_claim)
        trep, mt = agent_specialist_time_qualifiers(reference_truth, system_claim)
        specialists = [nrep, erep, trep]
        meta["numbers"] = mn
        meta["entities"] = me
        meta["time_qualifiers"] = mt

    arb, ma = agent_arbiter(reference_truth, system_claim, suff, specialists)
    meta["arbiter"] = ma

    return CommitteeResult(
        row_index=row_index,
        reference_truth=reference_truth,
        system_claim=system_claim,
        sufficiency=suff,
        specialists=specialists,
        arbiter=arb,
        meta=meta,
    )


# =====================================================
# Pretty Output
# =====================================================

def _clip(s: str, n: int = 900) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def _fmt_tokens(meta: Dict[str, Any]) -> str:
    u = meta.get("usage", {}) if meta else {}
    it = u.get("input_tokens")
    ot = u.get("output_tokens")
    if it is None or ot is None:
        return ""
    return f" (toks {it}/{ot})"

def print_result(res: CommitteeResult) -> None:
    print("\n" + "=" * 80)
    print(f"ROW {res.row_index} — FINAL: {res.arbiter.final_verdict} (conf={res.arbiter.confidence:.2f})")
    print("=" * 80)

    print("\nREFERENCE TRUTH:")
    print(_clip(res.reference_truth))
    print("\nSYSTEM CLAIM:")
    print(_clip(res.system_claim))

    print("\n--- CONTEXT SUFFICIENCY ---")
    print(f"sufficient={res.sufficiency.sufficient} (conf={res.sufficiency.confidence:.2f})")
    print(_clip(res.sufficiency.reasoning, 700))
    if res.sufficiency.key_dimensions:
        print("key_dimensions:")
        for d in res.sufficiency.key_dimensions[:10]:
            print(f"- {d}")
    if not res.sufficiency.sufficient and res.sufficiency.needed_info:
        print("needed_info:")
        for x in res.sufficiency.needed_info[:12]:
            print(f"- {x}")

    if res.specialists:
        print("\n--- SPECIALISTS ---")
        for s in res.specialists:
            status = "PASS" if s.passed else "FAIL"
            print(f"[{s.specialist}] {status} (conf={s.confidence:.2f})")
            print(f"  {_clip(s.reasoning, 450)}")
            for iss in s.issues[:6]:
                print(f"  - issue: {_clip(iss, 240)}")

    print("\n--- ARBITER ---")
    print(_clip(res.arbiter.summary_reason, 700))
    for p in res.arbiter.decisive_points[:10]:
        print(f"- {_clip(p, 260)}")
    if res.arbiter.final_verdict == "AMBIGUOUS" and res.arbiter.required_extra_info:
        print("required_extra_info:")
        for x in res.arbiter.required_extra_info[:12]:
            print(f"- {_clip(x, 240)}")

    print("\n--- PERF ---")
    for k, m in res.meta.items():
        print(f"{k:18s}: {m.get('elapsed', 0.0):.2f}s{_fmt_tokens(m)}")


# =====================================================
# Main
# =====================================================

def main():
    df = pd.read_csv("North_Star.csv")

    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    rows_to_check = [5, 6]  # only two rows as requested
    SWAP_FACT_AND_CLAIM = False

    rows = df.iloc[[i for i in rows_to_check if i in df.index]]

    print("\nFACTTRACE — NO-WEB CONTEXT SUFFICIENCY + SPECIALIST JURY (2 rows)")
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
