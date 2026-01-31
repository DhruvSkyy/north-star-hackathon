import json
import pandas as pd

from facttrace import run_committee


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
        for agent in result.agent_outputs:
            print(f"\n[{agent.agent_name}] (confidence={agent.confidence:.2f})")
            print(json.dumps(agent.output, indent=2, ensure_ascii=False))

        if result.critic:
            print("\n--- CRITIC ---")
            print(f"(confidence={result.critic.confidence:.2f})")
            print(json.dumps(result.critic.model_dump(), indent=2, ensure_ascii=False))

        if result.reruns:
            print("\nRERUNS TRIGGERED:", result.reruns)

        print("\n=== FINAL VERDICT ===")
        print(result.final_verdict)
        print("WHY:", result.summary_reason)
        print("SCORE:", {k: round(v, 3) for k, v in result.score.items()})
        print("\n")


if __name__ == "__main__":
    main()
