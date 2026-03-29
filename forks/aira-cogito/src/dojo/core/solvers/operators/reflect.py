# Copyright (c) 2026 Research Society
# MC-ESES: reflect_op — the state update function U(z_t, r_t) → z_{t+1}
#
# After executing code and obtaining feedback, this operator asks the LLM
# to update the cognitive state: revise hypotheses, record learned patterns,
# adjust search preferences and confidence.

import logging
from typing import Optional

from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.cognitive_state import CognitiveState
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import wrap_code
from dojo.utils.code_parsing import parse_json_output

log = logging.getLogger(__name__)

# JSON schema for the structured reflect response
REFLECT_SCHEMA = """{
    "type": "object",
    "properties": {
        "task_understanding": {
            "type": "string",
            "description": "Updated, refined understanding of the task based on all evidence so far."
        },
        "hypotheses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of active working hypotheses about what approaches will work. Drop disproved ones, add new ones."
        },
        "learned_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Patterns learned from execution feedback (e.g. 'batch size > 64 causes OOM', 'feature X is highly predictive')."
        },
        "preferred_directions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Directions to explore next, ranked by promise."
        },
        "avoided_directions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Dead ends to avoid in future attempts."
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in current best approach (0.0 = no idea, 1.0 = very confident). Be honest."
        },
        "key_insight": {
            "type": "string",
            "description": "The single most important insight from this latest attempt."
        }
    },
    "required": ["task_understanding", "hypotheses", "learned_patterns",
                  "preferred_directions", "avoided_directions", "confidence",
                  "key_insight"]
}"""


def reflect_op(
    reflect_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    current_state: CognitiveState,
    node: Node,
    journal: Journal,
) -> tuple:
    """State update function: z_{t+1} = U(z_t, r_t).

    Takes the current cognitive state and rich feedback from the latest
    code execution, and produces an updated cognitive state via LLM.

    Returns
    -------
    (updated_state, metrics) : tuple[CognitiveState, dict]
        The evolved cognitive state and LLM usage metrics.
    """
    # Assemble the feedback signal r_t
    metric_str = "N/A (buggy)" if node.is_buggy else str(node.metric.value)
    analysis_str = node.analysis or "(no analysis available)"

    # Build best-so-far context
    best_node = journal.get_best_node()
    if best_node and not best_node.metric.is_worst:
        best_metric_str = f"Best score so far: {best_node.metric.value} (step {best_node.step})"
    else:
        best_metric_str = "No successful submission yet."

    reflect_data = {
        "task_desc": task_description,
        "cognitive_state": current_state.to_prompt_str(),
        "prev_code": wrap_code(node.code),
        "prev_terminal_output": wrap_code(node.term_out, lang=""),
        "prev_analysis": analysis_str,
        "prev_metric": metric_str,
        "is_buggy": str(node.is_buggy),
        "best_so_far": best_metric_str,
        "evolution_step": str(current_state.evolution_step),
    }

    response_text, metrics = reflect_llm(
        query_data=reflect_data,
        json_schema=REFLECT_SCHEMA,
        function_name="update_cognitive_state",
        function_description="Update the agent's cognitive state based on the latest execution feedback.",
        no_user_message=True,
    )

    # Parse LLM response
    parsed = parse_json_output(response_text)
    if not isinstance(parsed, dict) or "task_understanding" not in parsed:
        log.warning("reflect_op: malformed LLM response, keeping previous state")
        # Still record the attempt
        current_state.add_attempt(
            step=current_state.evolution_step,
            approach=node.plan or "(unknown)",
            metric=node.metric.value if hasattr(node.metric, "value") and node.metric.value is not None else None,
            is_buggy=node.is_buggy,
            key_insight="(reflection failed)",
        )
        current_state.evolution_step += 1
        return current_state, metrics

    # Build the updated state
    updated = CognitiveState(
        task_understanding=parsed.get("task_understanding", current_state.task_understanding),
        hypotheses=parsed.get("hypotheses", current_state.hypotheses),
        learned_patterns=parsed.get("learned_patterns", current_state.learned_patterns),
        preferred_directions=parsed.get("preferred_directions", current_state.preferred_directions),
        avoided_directions=parsed.get("avoided_directions", current_state.avoided_directions),
        confidence=float(parsed.get("confidence", current_state.confidence)),
        # Carry forward immutable context
        environment_context=current_state.environment_context,
        # Carry forward history
        attempt_summaries=list(current_state.attempt_summaries),
        evolution_step=current_state.evolution_step + 1,
    )

    # Record this attempt
    updated.add_attempt(
        step=current_state.evolution_step,
        approach=node.plan or "(unknown)",
        metric=node.metric.value if hasattr(node.metric, "value") and node.metric.value is not None else None,
        is_buggy=node.is_buggy,
        key_insight=parsed.get("key_insight", ""),
    )

    return updated, metrics
