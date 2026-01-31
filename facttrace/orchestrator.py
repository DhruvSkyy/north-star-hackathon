"""Orchestrator that wires agents together."""

from typing import List

from .agents import (
    ContextBuilderAgent,
    ContextGapDetectorAgent,
    FactCheckerAgent,
    IntegrityValidatorAgent,
    RelevancyAssessorAgent,
)
from .arbiter import arbitrate
from .schemas import AgentVerdict, JuryResult


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


__all__ = ["run_committee"]
