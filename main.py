import os
import json
import time
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Environment & OpenAI
# =========================

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")  # put it in .env, do NOT hard-code it
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)

MODEL = "gpt-5.2"

# Pricing per 1M tokens (input, output)
PRICING = {
    "gpt-5.2": (1.75, 14.00),
}

# =========================
# Data Models
# =========================

class AgentVerdict(BaseModel):
    agent_name: str
    verdict: str = Field(..., description="faithful or mutated")
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class JuryResult(BaseModel):
    final_verdict: str
    summary_reason: str
    agent_votes: List[AgentVerdict]

# =========================
# OpenAI Call Helper
# =========================

def call_openai(prompt: str) -> dict:
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    elapsed = time.perf_counter() - start
    usage = response.usage
    content = response.choices[0].message.content

    in_price, out_price = PRICING[MODEL]
    input_cost = usage.prompt_tokens * in_price / 1_000_000
    output_cost = usage.completion_tokens * out_price / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "content": content,
        "elapsed": elapsed,
        "usage": usage,
        "cost": total_cost,
    }

# =========================
# Jury Agent
# =========================

class JuryAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def judge(self, reference_truth: str, system_claim: str) -> AgentVerdict:
        prompt = f"""
You are a member of an AI truth jury.

AGENT NAME:
{self.name}

ROLE:
{self.role}

REFERENCE TRUTH (ground truth):
"{reference_truth}"

SYSTEM CLAIM (derived statement):
"{system_claim}"

TASK:
- Decide whether the SYSTEM CLAIM is a FAITHFUL or MUTATED representation of the REFERENCE TRUTH.
- Focus on meaning, implication, scope, qualifiers, and causal framing.
- "Technically true" is NOT sufficient if the meaning shifts.

Return ONLY valid JSON in this exact format:
{{
  "verdict": "faithful or mutated",
  "reasoning": "clear explanation",
  "confidence": 0.0
}}
"""
        result = call_openai(prompt)
        content = result["content"]

        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]

        parsed = json.loads(content)

        return AgentVerdict(
            agent_name=self.name,
            verdict=parsed["verdict"].lower(),
            reasoning=parsed["reasoning"],
            confidence=float(parsed["confidence"]),
        )

# =========================
# Arbiter (Non-LLM)
# =========================

def arbitrate(votes: List[AgentVerdict]) -> JuryResult:
    mutated = [v for v in votes if v.verdict == "mutated"]
    faithful = [v for v in votes if v.verdict == "faithful"]

    if len(mutated) >= 2:
        final = "MUTATED"
        reason = "Multiple agents identified semantic drift, exaggeration, or misleading framing."
    elif len(faithful) == len(votes):
        final = "FAITHFUL"
        reason = "All agents agree the claim preserves the original meaning."
    else:
        final = "AMBIGUOUS"
        reason = "Agents disagreed; the claim is borderline and interpretation-dependent."

    return JuryResult(final_verdict=final, summary_reason=reason, agent_votes=votes)

# =========================
# Run Jury on One Pair
# =========================

def run_jury(reference_truth: str, system_claim: str) -> JuryResult:
    agents = [
        JuryAgent("Literal Fact Checker", "Be pedantic. Check numbers, qualifiers, scope, and literal accuracy."),
        JuryAgent("Context & Intent Analyst", "Assess whether the overall meaning or implication has shifted."),
        JuryAgent("Sceptic", "Assume the claim is misleading. Actively try to prove mutation."),
        JuryAgent("Common Sense Judge", "Judge whether a reasonable reader would be misled."),
    ]

    votes = [agent.judge(reference_truth, system_claim) for agent in agents]
    return arbitrate(votes)

# =========================
# Main
# =========================

def main():
    df = pd.read_csv("North_Star.csv")

    # Columns in your CSV
    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    # Pick EXACTLY the rows you want (these should be the 5 “interesting” ones)
    # (Example indices — replace with your chosen five.)
    rows_to_check = [5, 6, 7, 12, 19]

    # If you suspect the dataset columns are reversed, flip this to True.
    SWAP_FACT_AND_CLAIM = False

    # Subset the dataframe to ONLY those rows (and drop any out-of-range)
    valid_rows = [i for i in rows_to_check if 0 <= i < len(df)]
    sub = df.iloc[valid_rows].copy()

    print("\n====================================")
    print(" FACTTRACE – AGENTIC CONSENSUS (GPT-5.2)")
    print("====================================\n")

    for idx, row in sub.iterrows():
        truth_text = str(row[TRUTH_COL])
        claim_text = str(row[CLAIM_COL])

        # Direction control (fixes your “swapped around” concern)
        if SWAP_FACT_AND_CLAIM:
            reference_truth, system_claim = claim_text, truth_text
        else:
            reference_truth, system_claim = truth_text, claim_text

        print("\n------------------------------------")
        print(f"ROW {idx}")
        print("------------------------------------")
        print("\nREFERENCE TRUTH:")
        print(reference_truth)
        print("\nSYSTEM CLAIM:")
        print(system_claim)

        result = run_jury(reference_truth, system_claim)

        print("\n--- AGENT DEBATE ---")
        for v in result.agent_votes:
            print(f"\n[{v.agent_name}]")
            print(f"Verdict   : {v.verdict.upper()}")
            print(f"Confidence: {v.confidence}")
            print(f"Reasoning : {v.reasoning}")

        print("\n=== FINAL VERDICT ===")
        print(result.final_verdict)
        print("WHY:", result.summary_reason)
        print("\n")

if __name__ == "__main__":
    main()
