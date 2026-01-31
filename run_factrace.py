# run_facttrace.py
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from facttrace_jury import run_committee_one, print_result

def main():
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Model:
    # - If you want best tool-use / reasoning, consider newer tool-capable models.
    # - Keeping your original default here.
    MODEL = os.getenv("FACTTRACE_MODEL", "gpt-4.1")

    df = pd.read_csv("North_Star.csv")
    TRUTH_COL = "truth"
    CLAIM_COL = "claim"

    # Only two rows as requested:
    rows_to_check = [5, 6]

    # If you need to swap:
    SWAP_FACT_AND_CLAIM = False

    rows = df.iloc[[i for i in rows_to_check if i in df.index]]

    print("\nFACTTRACE — WEB-CONTEXT JURY (2 rows)")
    print(f"Model: {MODEL}")
    print("-" * 72)

    for idx, row in rows.iterrows():
        truth = str(row[TRUTH_COL])
        claim = str(row[CLAIM_COL])

        reference_truth, system_claim = (claim, truth) if SWAP_FACT_AND_CLAIM else (truth, claim)

        result = run_committee_one(
            client,
            model=MODEL,
            row_index=int(idx),
            reference_truth=reference_truth,
            system_claim=system_claim,
            # low/medium/high: how much web context to include
            search_context_size="medium",
        )

        print_result(result)

if __name__ == "__main__":
    main()
