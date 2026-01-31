import os
import json
import time
import pandas as pd
from typing import List
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import OpenAI, APIError

# =========================
# Environment & OpenAI
# =========================

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)

MODEL = "gpt-4.1"   # explicit 4.1 usage
TEMPERATURE = 0.0   # determinism for jury work
MAX_RETRIES = 3

PRICING = {
    "gpt-4.1": (1.00, 3.00),  # update if needed
}

# =========================
# Data Models
# =========================

class AgentVerdict(BaseModel):
    agent_name: str
    verdict: str = Field(..., pattern="^(faithful|mutated)$")
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class JuryResult(BaseModel):
    final_verdict: str
    summary_reason: str
    agent_votes: List[AgentVerdict]

# =========================
# OpenAI Call Helper (4.1)
# =========================

def call_openai_json(prompt: str) -> dict:
    """Robust JSON-only OpenAI call with retries and cost tracking."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.perf_counter()

            response = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )

            elapsed = time.perf_counter() - start
            usage = response.usage
            content = response.output_text

            in_price, out_price = PRICING[MODEL]
            cost = (
                usage.input_tokens * in_price
                + usage.output_tokens * out_price
            ) / 1_000_000

            return {
                "content": content,
                "elapsed": elapsed,
                "usage": usage,
                "cost": cost,
            }

        except (APIError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"OpenAI call failed after retries: {last_error}")

# =========================
# Jury Agent
# =========================

class JuryAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def judge(self, reference_truth: str, system_claim: str) -> AgentVerdict:
        prompt = f"""
You are an AI truth jury member.

AGENT NAME:
{self.name}

ROLE:
{self.role}

REFERENCE TRUTH:
{reference_truth}

SYSTEM CLAIM:
{system_claim}

DECISION RULES:
- Decide if the claim is FAITHFUL or MUTATED relative to the truth.
- Semantic drift, exaggeration, missing qualifiers = MUTATED.
- Literal correctness alone is insufficient.

Return valid JSON ONLY:

{{
  "verdict": "faithful or mutated",
  "reasoning": "concise but precise explanation",
  "confidence": 0.0
}}
"""

        result = call_openai_json(prompt)

        try:
            parsed = json.loads(result["content"])
            return AgentVerdict(
                agent_name=self.name,
                verdict=parsed["verdict"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed["confidence"]),
            )
        except (KeyError, ValidationError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid agent output from {self.name}: {e}")

# =========================
# Arbiter (Confidence-Weighted)
# =========================

def arbitrate(votes: List[AgentVerdict]) -> JuryResult:
    score = {"faithful": 0.0, "mutated": 0.0}

    for v in votes:
        score[v.verdict] += v.confidence

    if score["mutated"] >= score["faithful"] + 0.5:
        final = "MUTATED"
        reason = "Confidence-weighted consensus indicates semantic distortion."
    elif score["faithful"] >= score["mutated"] + 0.5:
        final = "FAITHFUL"
        reason = "Confidence-weighted consensus indicates meaning preserved."
    else:
        final = "AMBIGUOUS"
        reason = "Confidence scores are close; interpretation-dependent."

    return JuryResult(
        final_verdict=final,
        summary_reason=reason,
        agent_votes=votes,
    )

# =========================
# Run Jury
# =========================

def run_jury(reference_truth: str, system_claim: str) -> JuryResult:
    agents = [
        JuryAgent(
            "Literal Fact Checker",
            "Check numerical accuracy, qualifiers, and explicit scope."
        ),
        JuryAgent(
            "Context & Intent Analyst",
            "Evaluate implied meaning and framing."
        ),
        JuryAgent(
            "Sceptic",
            "Assume the claim is misleading unless proven otherwise."
        ),
        JuryAgent(
            "Common Sense Judge",
            "Assess how a reasonable reader would interpret it."
        ),
    ]

    votes = [agent.judge(reference_truth, system_claim) for agent in agents]
    return arbitrate(votes)

# =========================
# Main
# =========================

def main():
    df = pd.read_csv("North_Star.csv")

    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    rows_to_check = [5, 6, 7, 12, 19]
    SWAP_FACT_AND_CLAIM = False

    rows = df.iloc[[i for i in rows_to_check if i in df.index]]

    print("\n====================================")
    print(" FACTTRACE – AGENTIC CONSENSUS (GPT-4.1)")
    print("====================================\n")

    for idx, row in rows.iterrows():
        truth = str(row[TRUTH_COL])
        claim = str(row[CLAIM_COL])

        reference_truth, system_claim = (
            (claim, truth) if SWAP_FACT_AND_CLAIM else (truth, claim)
        )

        print(f"\n--- ROW {idx} ---")
        print("\nREFERENCE TRUTH:\n", reference_truth)
        print("\nSYSTEM CLAIM:\n", system_claim)

        result = run_jury(reference_truth, system_claim)

        print("\n--- AGENT VERDICTS ---")
        for v in result.agent_votes:
            print(
                f"\n[{v.agent_name}] "
                f"{v.verdict.upper()} "
                f"(conf={v.confidence:.2f})\n"
                f"{v.reasoning}"
            )

        print("\n=== FINAL VERDICT ===")
        print(result.final_verdict)
        print("WHY:", result.summary_reason)

if __name__ == "__main__":
    main()
