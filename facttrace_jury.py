# facttrace_jury.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# -----------------------------
# Types / Models
# -----------------------------

VerdictLabel = Literal["faithful", "mutated", "ambiguous"]
FinalVerdict = Literal["FAITHFUL", "MUTATED", "AMBIGUOUS"]

class GapAnalysis(BaseModel):
    missing_context_questions: List[str] = Field(default_factory=list)
    key_dimensions: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class WebSource(BaseModel):
    title: str
    url: str
    snippet: str

class BuiltContext(BaseModel):
    distilled_reference: str
    definitions: List[str] = Field(default_factory=list)
    scope_rules: List[str] = Field(default_factory=list)
    qualifier_rules: List[str] = Field(default_factory=list)

    # Added: evidence from the web-search step
    web_answers: Dict[str, str] = Field(default_factory=dict, description="question -> answer")
    sources: List[WebSource] = Field(default_factory=list)

    confidence: float = Field(..., ge=0.0, le=1.0)

class ArgumentReport(BaseModel):
    stance: Literal["faithful_advocate", "mutated_advocate"]
    verdict: VerdictLabel
    reasoning: str
    key_points: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class JudgeReport(BaseModel):
    final_verdict: FinalVerdict
    summary_reason: str
    decisive_points: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

@dataclass
class CallMeta:
    elapsed_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    raw_json: Dict[str, Any]

@dataclass
class CommitteeResult:
    row_index: int
    reference_truth: str
    system_claim: str
    gap: GapAnalysis
    context: BuiltContext
    faithful_advocate: ArgumentReport
    mutated_advocate: ArgumentReport
    judge: JudgeReport
    metas: Dict[str, CallMeta]


# -----------------------------
# OpenAI Helpers
# -----------------------------

def _now() -> float:
    return time.perf_counter()

def _usage_tokens(resp: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return None, None
    return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)

def call_parse(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    out_model: Any,  # pydantic model class
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
) -> Tuple[Any, CallMeta]:
    """
    Uses Responses API structured parsing when available in the SDK.
    If your SDK doesn’t support responses.parse, replace with responses.create + json_object format.
    """
    start = _now()
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tools or [],
        temperature=temperature,
        text_format=out_model,
    )
    elapsed = _now() - start
    inp_toks, out_toks = _usage_tokens(resp)
    parsed = resp.output_parsed

    meta = CallMeta(
        elapsed_s=elapsed,
        input_tokens=inp_toks,
        output_tokens=out_toks,
        raw_json={"output": getattr(resp, "output", None)},
    )
    return parsed, meta


# -----------------------------
# Agents
# -----------------------------

class ContextGapDetectorAgent:
    name = "ContextGapDetector"

    @staticmethod
    def run(client: OpenAI, *, model: str, reference_truth: str, system_claim: str) -> Tuple[GapAnalysis, CallMeta]:
        system = (
            "You are a specialised context gap detector for truthfulness evaluation.\n"
            "Identify what context is REQUIRED to judge whether the claim faithfully preserves the meaning of the reference.\n"
            "Be strict about qualifiers, scope, timeframe, entity identity, definitions.\n"
            "Return JSON only."
        )
        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

TASK (JSON):
- missing_context_questions: minimum questions needed to judge faithfulness WITHOUT guessing
- key_dimensions: key audit dimensions (scope, timeframe, entity, definitions, qualifiers, causality, units, modality)
- confidence: 0..1
"""
        return call_parse(
            client,
            model=model,
            system=system,
            user=user,
            out_model=GapAnalysis,
            tools=None,
            temperature=0.0,
        )


class WebContextBuilderAgent:
    """
    ONLY agent allowed to use web_search.
    It answers the gap questions using web search, then builds a context bundle used by the debaters/judge.
    """
    name = "WebContextBuilder"

    @staticmethod
    def run(
        client: OpenAI,
        *,
        model: str,
        reference_truth: str,
        gap: GapAnalysis,
        search_context_size: Literal["low", "medium", "high"] = "medium",
    ) -> Tuple[BuiltContext, CallMeta]:
        system = (
            "You are a specialised context builder for semantic faithfulness checking.\n"
            "You may use WEB SEARCH to answer missing-context questions.\n"
            "Rules:\n"
            "- Use the REFERENCE TRUTH as the anchor (do not rewrite it into a different claim).\n"
            "- Use web search ONLY to answer the missing-context questions (definitions, identity, timeframe, background facts).\n"
            "- If web evidence is inconclusive, say so.\n"
            "- Provide sources (title, url, snippet).\n"
            "Return JSON only."
        )

        questions = "\n".join(f"- {q}" for q in gap.missing_context_questions) or "- (none)"
        dims = "\n".join(f"- {d}" for d in gap.key_dimensions) or "- (none)"

        user = f"""
REFERENCE TRUTH:
{reference_truth}

MISSING CONTEXT QUESTIONS:
{questions}

KEY DIMENSIONS:
{dims}

OUTPUT JSON:
- distilled_reference: short neutral distillation of reference_truth
- definitions: any important definitions
- scope_rules: what is in/out of scope for judging this
- qualifier_rules: qualifiers that must be preserved
- web_answers: map question -> answer (based on web evidence)
- sources: list of sources used (title, url, snippet)
- confidence: 0..1
"""

        # Hosted tool: web_search. :contentReference[oaicite:1]{index=1}
        tools = [
            {
                "type": "web_search"
            }
        ]

        return call_parse(
            client,
            model=model,
            system=system,
            user=user,
            out_model=BuiltContext,
            tools=tools,
            temperature=0.0,
        )


class DebateAgent:
    """
    Two stances:
      - faithful_advocate: argue it's faithful unless clearly not
      - mutated_advocate: argue it's mutated unless clearly faithful
    No web access: must use BuiltContext only.
    """
    name = "DebateAgent"

    @staticmethod
    def run(
        client: OpenAI,
        *,
        model: str,
        stance: Literal["faithful_advocate", "mutated_advocate"],
        reference_truth: str,
        system_claim: str,
        context: BuiltContext,
    ) -> Tuple[ArgumentReport, CallMeta]:
        system = (
            "You are a truthfulness debater.\n"
            "You do NOT have web access. Use ONLY the provided context bundle.\n"
            "Your job is to argue your assigned stance rigorously.\n"
            "If you must concede uncertainty, mark verdict=ambiguous.\n"
            "Return JSON only."
        )

        stance_instructions = {
            "faithful_advocate": (
                "STANCE: Faithful advocate.\n"
                "Argue the claim is faithful to the reference.\n"
                "You should still flag any real qualifier/scope drift; do NOT handwave."
            ),
            "mutated_advocate": (
                "STANCE: Mutation advocate.\n"
                "Argue the claim is mutated (semantic drift / missing qualifiers / scope shift / added unsupported info).\n"
                "Be strict and point out subtle meaning changes."
            ),
        }[stance]

        user = f"""
{stance_instructions}

REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

CONTEXT BUNDLE:
distilled_reference: {context.distilled_reference}
definitions: {context.definitions}
scope_rules: {context.scope_rules}
qualifier_rules: {context.qualifier_rules}
web_answers: {context.web_answers}

OUTPUT JSON:
- stance: "{stance}"
- verdict: faithful|mutated|ambiguous
- reasoning: short but concrete
- key_points: bullet points
- confidence: 0..1
"""

        return call_parse(
            client,
            model=model,
            system=system,
            user=user,
            out_model=ArgumentReport,
            tools=None,
            temperature=0.0,
        )


class JudgeAgent:
    """
    Final decision maker. No web access. Must decide based on:
      - context bundle
      - both arguments
    """
    name = "Judge"

    @staticmethod
    def run(
        client: OpenAI,
        *,
        model: str,
        reference_truth: str,
        system_claim: str,
        context: BuiltContext,
        a_faithful: ArgumentReport,
        a_mutated: ArgumentReport,
    ) -> Tuple[JudgeReport, CallMeta]:
        system = (
            "You are an impartial judge deciding semantic faithfulness.\n"
            "You do NOT have web access. Use ONLY the provided context + arguments.\n"
            "Decision rules:\n"
            "- FAITHFUL only if meaning is preserved with no material qualifier/scope/timeframe/implication drift.\n"
            "- MUTATED if any material drift or unsupported additions.\n"
            "- AMBIGUOUS if context evidence is insufficient to decide.\n"
            "Return JSON only."
        )

        user = f"""
REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

CONTEXT BUNDLE (authoritative for this decision):
distilled_reference: {context.distilled_reference}
definitions: {context.definitions}
scope_rules: {context.scope_rules}
qualifier_rules: {context.qualifier_rules}
web_answers: {context.web_answers}

ARGUMENTS:
FAITHFUL ADVOCATE:
- verdict: {a_faithful.verdict}
- reasoning: {a_faithful.reasoning}
- key_points: {a_faithful.key_points}
- confidence: {a_faithful.confidence}

MUTATION ADVOCATE:
- verdict: {a_mutated.verdict}
- reasoning: {a_mutated.reasoning}
- key_points: {a_mutated.key_points}
- confidence: {a_mutated.confidence}

OUTPUT JSON:
- final_verdict: FAITHFUL|MUTATED|AMBIGUOUS
- summary_reason: short explanation
- decisive_points: list of the deciding considerations
- confidence: 0..1
"""
        return call_parse(
            client,
            model=model,
            system=system,
            user=user,
            out_model=JudgeReport,
            tools=None,
            temperature=0.0,
        )


# -----------------------------
# Orchestrator
# -----------------------------

def run_committee_one(
    client: OpenAI,
    *,
    model: str,
    row_index: int,
    reference_truth: str,
    system_claim: str,
    search_context_size: Literal["low", "medium", "high"] = "medium",
) -> CommitteeResult:
    metas: Dict[str, CallMeta] = {}

    gap, m1 = ContextGapDetectorAgent.run(
        client, model=model, reference_truth=reference_truth, system_claim=system_claim
    )
    metas["gap"] = m1

    context, m2 = WebContextBuilderAgent.run(
        client,
        model=model,
        reference_truth=reference_truth,
        gap=gap,
        search_context_size=search_context_size,
    )
    metas["context"] = m2

    faithful, m3 = DebateAgent.run(
        client,
        model=model,
        stance="faithful_advocate",
        reference_truth=reference_truth,
        system_claim=system_claim,
        context=context,
    )
    metas["faithful_advocate"] = m3

    mutated, m4 = DebateAgent.run(
        client,
        model=model,
        stance="mutated_advocate",
        reference_truth=reference_truth,
        system_claim=system_claim,
        context=context,
    )
    metas["mutated_advocate"] = m4

    judge, m5 = JudgeAgent.run(
        client,
        model=model,
        reference_truth=reference_truth,
        system_claim=system_claim,
        context=context,
        a_faithful=faithful,
        a_mutated=mutated,
    )
    metas["judge"] = m5

    return CommitteeResult(
        row_index=row_index,
        reference_truth=reference_truth,
        system_claim=system_claim,
        gap=gap,
        context=context,
        faithful_advocate=faithful,
        mutated_advocate=mutated,
        judge=judge,
        metas=metas,
    )


# -----------------------------
# Pretty Printing
# -----------------------------

def _clip(s: str, n: int = 700) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def print_result(result: CommitteeResult) -> None:
    print("\n" + "=" * 72)
    print(f"ROW {result.row_index} — FINAL: {result.judge.final_verdict}  (conf={result.judge.confidence:.2f})")
    print("=" * 72)

    print("\nREFERENCE TRUTH:")
    print(_clip(result.reference_truth, 900))
    print("\nSYSTEM CLAIM:")
    print(_clip(result.system_claim, 900))

    print("\n--- GAP QUESTIONS ---")
    for q in result.gap.missing_context_questions[:12]:
        print(f"- {q}")
    if len(result.gap.missing_context_questions) > 12:
        print(f"... (+{len(result.gap.missing_context_questions) - 12} more)")

    print("\n--- CONTEXT (DISTILLED) ---")
    print(result.context.distilled_reference)

    if result.context.web_answers:
        print("\n--- WEB ANSWERS ---")
        for k, v in list(result.context.web_answers.items())[:10]:
            print(f"* Q: {k}\n  A: {v}\n")
        if len(result.context.web_answers) > 10:
            print(f"... (+{len(result.context.web_answers) - 10} more)")

    if result.context.sources:
        print("\n--- SOURCES USED (TOP) ---")
        for src in result.context.sources[:6]:
            print(f"- {src.title}\n  {src.url}\n  {src.snippet}\n")
        if len(result.context.sources) > 6:
            print(f"... (+{len(result.context.sources) - 6} more)")

    print("\n--- DEBATE ---")
    print(f"[Faithful advocate] verdict={result.faithful_advocate.verdict} conf={result.faithful_advocate.confidence:.2f}")
    for p in result.faithful_advocate.key_points[:6]:
        print(f"  - {p}")

    print(f"\n[Mutation advocate] verdict={result.mutated_advocate.verdict} conf={result.mutated_advocate.confidence:.2f}")
    for p in result.mutated_advocate.key_points[:6]:
        print(f"  - {p}")

    print("\n--- JUDGE ---")
    print(result.judge.summary_reason)
    for p in result.judge.decisive_points[:8]:
        print(f"- {p}")

    # small perf summary
    print("\n--- RUNTIME / TOKENS ---")
    for k, meta in result.metas.items():
        toks = ""
        if meta.input_tokens is not None and meta.output_tokens is not None:
            toks = f" | toks in/out={meta.input_tokens}/{meta.output_tokens}"
        print(f"{k:16s}: {meta.elapsed_s:.2f}s{toks}")
