"""Deterministic arbiter logic."""

from typing import Dict, Tuple, Optional

from .schemas import (
    BuiltContext,
    CriticReport,
    FactCheckReport,
    GapAnalysis,
    IntegrityReport,
    RelevancyReport,
)


def arbitrate(
    relevancy: RelevancyReport,
    factcheck: FactCheckReport,
    integrity: IntegrityReport,
    context: BuiltContext,
    gap: GapAnalysis,
    critic: Optional[CriticReport] = None,
) -> Tuple[str, str, Dict[str, float]]:
    """
    Deterministic decision logic inspired by the paper’s rule-based arbiter:
    - Reject (mutated) if contradiction/qualifier loss/scope shift/etc. are confidently present.
    - Faithful only if no critical flags and relevancy is true and integrity passes.
    - Otherwise ambiguous.
    """

    score = {"faithful": 0.0, "mutated": 0.0, "ambiguous": 0.0}
    critic_note = ""

    if not relevancy.relevant and relevancy.confidence >= 0.70:
        score["mutated"] += 1.0
        return (
            "MUTATED",
            "Claim appears topically/entity-wise irrelevant to the reference.",
            score,
        )

    if not integrity.passes and integrity.confidence >= 0.70:
        score["ambiguous"] += 1.0
        return (
            "AMBIGUOUS",
            "Integrity validator found internal inconsistencies in the fact-check report.",
            score,
        )

    critical_flags = [
        ("contradicts_reference", 1.2),
        ("adds_new_unsupported_info", 0.9),
        ("drops_required_qualifiers", 1.1),
        ("shifts_scope_or_timeframe", 1.0),
        ("changes_implication_or_intent", 1.1),
    ]

    mutated_weight = 0.0
    for flag, weight in critical_flags:
        if getattr(factcheck, flag):
            mutated_weight += weight

    if factcheck.ambiguous_due_to_missing_context:
        score["ambiguous"] += 0.8 * factcheck.confidence

    if critic:
        if critic.issues:
            score["ambiguous"] += 0.4 * critic.confidence
            critic_note = " Critic flagged unresolved issues."
        if any(req.priority == "high" for req in critic.requests):
            score["ambiguous"] += 0.2 * critic.confidence

    score[factcheck.verdict] += 1.0 * factcheck.confidence
    score["mutated"] += mutated_weight * factcheck.confidence
    score["faithful"] += 0.15 * context.confidence
    score["ambiguous"] += 0.10 * (1.0 - context.confidence) + 0.05 * (1.0 - gap.confidence)

    sorted_scores = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_val = sorted_scores[0]
    second_val = sorted_scores[1][1]

    if top_val < second_val + 0.35:
        return (
            "AMBIGUOUS",
            "Scores are close; decision is interpretation-dependent given remaining uncertainty.",
            score,
        )

    if top_label == "mutated":
        return (
            "MUTATED",
            "Deterministic arbiter found high-confidence semantic distortion flags." + critic_note,
            score,
        )
    if top_label == "faithful":
        return (
            "FAITHFUL",
            "Deterministic arbiter found meaning preserved with no high-severity drift." + critic_note,
            score,
        )
    return ("AMBIGUOUS", "Arbiter could not reach a stable margin." + critic_note, score)


__all__ = ["arbitrate"]
