"""Orchestrator that wires agents together with a critic-driven rerun loop."""

from typing import Dict, List, Optional

from .agents import (
    ContextBuilderAgent,
    ContextGapDetectorAgent,
    CriticAgent,
    FactCheckerAgent,
    IntegrityValidatorAgent,
    RelevancyAssessorAgent,
)
from .arbiter import arbitrate
from .schemas import AgentVerdict, CriticReport, JuryResult


def run_committee(reference_truth: str, system_claim: str) -> JuryResult:
    gap_agent = ContextGapDetectorAgent()
    ctx_agent = ContextBuilderAgent()
    rel_agent = RelevancyAssessorAgent()
    fc_agent = FactCheckerAgent()
    iv_agent = IntegrityValidatorAgent()
    critic_agent = CriticAgent()

    rerun_limit = 1
    max_iterations = 2  # initial run + at most one rerun cycle
    rerun_counts: Dict[str, int] = {}
    feedbacks: Dict[str, str] = {}
    last_critic: Optional[CriticReport] = None
    agent_outputs: List[AgentVerdict] = []

    def execute_agents() -> tuple:
        """Run the full agent stack once using current feedback map."""
        local_outputs: List[AgentVerdict] = []

        gap = gap_agent.run(reference_truth, system_claim, critic_feedback=feedbacks.get(gap_agent.name))
        local_outputs.append(AgentVerdict(agent_name=gap_agent.name, output=gap.model_dump(), confidence=gap.confidence))

        context = ctx_agent.run(reference_truth, gap, critic_feedback=feedbacks.get(ctx_agent.name))
        local_outputs.append(AgentVerdict(agent_name=ctx_agent.name, output=context.model_dump(), confidence=context.confidence))

        relevancy = rel_agent.run(context, system_claim, critic_feedback=feedbacks.get(rel_agent.name))
        local_outputs.append(AgentVerdict(agent_name=rel_agent.name, output=relevancy.model_dump(), confidence=relevancy.confidence))

        factcheck = fc_agent.run(reference_truth, system_claim, context, critic_feedback=feedbacks.get(fc_agent.name))
        local_outputs.append(AgentVerdict(agent_name=fc_agent.name, output=factcheck.model_dump(), confidence=factcheck.confidence))

        integrity = iv_agent.run(factcheck, critic_feedback=feedbacks.get(iv_agent.name))
        local_outputs.append(AgentVerdict(agent_name=iv_agent.name, output=integrity.model_dump(), confidence=integrity.confidence))

        return gap, context, relevancy, factcheck, integrity, local_outputs

    for iteration in range(max_iterations):
        gap, context, relevancy, factcheck, integrity, agent_outputs = execute_agents()

        last_critic = critic_agent.run(reference_truth, system_claim, gap, context, relevancy, factcheck, integrity)

        # Decide on reruns
        if iteration == max_iterations - 1:
            break

        if last_critic.confidence < 0.5 or not last_critic.requests:
            break

        made_request = False
        for req in last_critic.requests:
            if rerun_counts.get(req.target_agent, 0) >= rerun_limit:
                continue
            rerun_counts[req.target_agent] = rerun_counts.get(req.target_agent, 0) + 1
            feedbacks[req.target_agent] = req.instruction
            made_request = True

        if not made_request:
            break
        # loop continues to rerun requested agents with feedback

    final_verdict, summary_reason, score = arbitrate(relevancy, factcheck, integrity, context, gap, critic=last_critic)

    return JuryResult(
        final_verdict=final_verdict,
        summary_reason=summary_reason,
        agent_outputs=agent_outputs,
        score=score,
        critic=last_critic,
        reruns=rerun_counts,
    )


__all__ = ["run_committee"]
