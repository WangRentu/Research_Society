# Copyright (c) 2026 Research Society
# Experiment instrumentation for "When Does Thinking Help?" paper.
#
# Provides:
#   - TriggerType: classification of what triggered a reflection
#   - classify_trigger(): determine trigger type from context
#   - compute_state_delta(): measure change between z_t and z_{t+1}
#   - save_trajectory_snapshot(): append one step to trajectory.jsonl

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.core.solvers.utils.cognitive_state import CognitiveState, Feedback


# ======================================================================
# Trigger classification (Exp 2)
# ======================================================================

class TriggerType(str, Enum):
    """What event triggered this reflection."""
    BOOTSTRAP = "bootstrap"          # first step, no history
    SCORE_JUMP = "score_jump"        # metric changed > 5%
    FIRST_SUCCESS = "first_success"  # first non-buggy result
    PLATEAU = "plateau"              # 3+ steps with no improvement
    NEW_ERROR = "new_error"          # new error category appeared
    ROUTINE = "routine"              # none of the above


def classify_trigger(
    feedback: Feedback,
    step: int,
    prev_error_categories: List[str],
    recent_metrics: List[Optional[float]],
    has_any_valid: bool,
) -> TriggerType:
    """Classify what triggered this reflection based on context.

    Args:
        feedback: current step's feedback r_t
        step: current evolution step (0-indexed)
        prev_error_categories: error categories seen in previous steps
        recent_metrics: last 3 valid metric values (oldest first)
        has_any_valid: whether any previous step produced a valid result
    """
    # Bootstrap: first step
    if step == 0:
        return TriggerType.BOOTSTRAP

    # First success: was buggy before, now valid
    if not feedback.is_buggy and not has_any_valid:
        return TriggerType.FIRST_SUCCESS

    # Score jump: metric changed > 5% relative to previous
    if feedback.metric is not None and recent_metrics:
        prev_valid = [m for m in recent_metrics if m is not None]
        if prev_valid:
            prev = prev_valid[-1]
            if prev != 0:
                rel_change = abs(feedback.metric - prev) / abs(prev)
                if rel_change > 0.05:
                    return TriggerType.SCORE_JUMP

    # New error type
    if feedback.is_buggy and feedback.error_category:
        if feedback.error_category not in prev_error_categories and feedback.error_category != "none":
            return TriggerType.NEW_ERROR

    # Plateau: last 3 metrics are all within 1% of each other
    if len(recent_metrics) >= 3:
        valid_recent = [m for m in recent_metrics[-3:] if m is not None]
        if len(valid_recent) >= 3:
            spread = max(valid_recent) - min(valid_recent)
            mean_val = sum(valid_recent) / len(valid_recent)
            if mean_val != 0 and abs(spread / mean_val) < 0.01:
                return TriggerType.PLATEAU

    return TriggerType.ROUTINE


# ======================================================================
# State delta measurement (Exp 1 & 2)
# ======================================================================

def compute_state_delta(z_before: CognitiveState, z_after: CognitiveState) -> Dict[str, Any]:
    """Measure the magnitude of change between two cognitive states.

    Returns a dict with per-field deltas.
    """
    hyp_before = set(z_before.hypotheses)
    hyp_after = set(z_after.hypotheses)
    pat_before = set(z_before.learned_patterns)
    pat_after = set(z_after.learned_patterns)
    pref_before = set(z_before.preferred_directions)
    pref_after = set(z_after.preferred_directions)
    avoid_before = set(z_before.avoided_directions)
    avoid_after = set(z_after.avoided_directions)

    return {
        "confidence_delta": z_after.confidence - z_before.confidence,
        "hypotheses_added": list(hyp_after - hyp_before),
        "hypotheses_removed": list(hyp_before - hyp_after),
        "patterns_added": list(pat_after - pat_before),
        "patterns_removed": list(pat_before - pat_after),
        "preferred_added": list(pref_after - pref_before),
        "preferred_removed": list(pref_before - pref_after),
        "avoided_added": list(avoid_after - avoid_before),
        "avoided_removed": list(avoid_before - avoid_after),
        "task_understanding_changed": z_before.task_understanding != z_after.task_understanding,
        "total_field_changes": (
            abs(z_after.confidence - z_before.confidence) > 0.01
        ) + (
            len(hyp_after ^ hyp_before) > 0
        ) + (
            len(pat_after ^ pat_before) > 0
        ) + (
            len(pref_after ^ pref_before) > 0
        ) + (
            len(avoid_after ^ avoid_before) > 0
        ) + (
            z_before.task_understanding != z_after.task_understanding
        ),
    }


# ======================================================================
# Trajectory snapshot (Exp 1, 2, 3)
# ======================================================================

def save_trajectory_snapshot(
    path: Path,
    step: int,
    z_before: CognitiveState,
    z_after: CognitiveState,
    feedback: Feedback,
    action: str,
    trigger_type: TriggerType,
    state_delta: Dict[str, Any],
    metric_value: Optional[float] = None,
    intervention_mode: str = "natural",
) -> None:
    """Append one trajectory snapshot to trajectory.jsonl."""
    snapshot = {
        "step": step,
        "timestamp": time.time(),
        "intervention_mode": intervention_mode,
        "action": action,
        "trigger_type": trigger_type.value,
        # z_t before reflect
        "z_before": {
            "confidence": z_before.confidence,
            "evolution_step": z_before.evolution_step,
            "hypotheses_count": len(z_before.hypotheses),
            "patterns_count": len(z_before.learned_patterns),
            "preferred_count": len(z_before.preferred_directions),
            "avoided_count": len(z_before.avoided_directions),
            "intrinsic_quality": z_before.intrinsic_quality(),
            "hypotheses": z_before.hypotheses[:10],
            "learned_patterns": z_before.learned_patterns[:10],
            "task_understanding_len": len(z_before.task_understanding),
        },
        # z_t+1 after reflect
        "z_after": {
            "confidence": z_after.confidence,
            "evolution_step": z_after.evolution_step,
            "hypotheses_count": len(z_after.hypotheses),
            "patterns_count": len(z_after.learned_patterns),
            "preferred_count": len(z_after.preferred_directions),
            "avoided_count": len(z_after.avoided_directions),
            "intrinsic_quality": z_after.intrinsic_quality(),
            "hypotheses": z_after.hypotheses[:10],
            "learned_patterns": z_after.learned_patterns[:10],
            "task_understanding_len": len(z_after.task_understanding),
        },
        # feedback r_t
        "feedback": {
            "metric": feedback.metric,
            "is_buggy": feedback.is_buggy,
            "trend": feedback.trend,
            "trend_delta": feedback.trend_delta,
            "novelty": feedback.novelty,
            "error_category": feedback.error_category,
            "free_energy_drop": feedback.free_energy_drop,
        },
        # state delta
        "state_delta": {
            "confidence_delta": state_delta["confidence_delta"],
            "total_field_changes": state_delta["total_field_changes"],
            "task_understanding_changed": state_delta["task_understanding_changed"],
            "hypotheses_added_count": len(state_delta["hypotheses_added"]),
            "hypotheses_removed_count": len(state_delta["hypotheses_removed"]),
            "patterns_added_count": len(state_delta["patterns_added"]),
        },
        # outcome
        "metric_value": metric_value,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
