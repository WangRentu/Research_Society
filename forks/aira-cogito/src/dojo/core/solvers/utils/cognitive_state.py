# Copyright (c) 2026 Research Society
# Monte Carlo Endogenous State Evolution Search (MC-ESES)
# Cognitive State: the persistent internal state z_t that encodes the agent's
# evolving understanding of a task. Solutions are projections of this state,
# not the primary search object.

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AttemptSummary:
    """Compressed record of a single attempt and its outcome."""

    step: int = 0
    approach: str = ""
    metric: Optional[float] = None
    is_buggy: bool = True
    key_insight: str = ""


@dataclass
class CognitiveState:
    """Persistent internal state z_t encoding the agent's understanding.

    This is the core object that distinguishes MC-ESES from vanilla search:
    instead of searching over code solutions directly, we maintain and evolve
    this structured representation of *what the agent knows*.  Solutions are
    generated as projections  a_t = E(z_t)  and feedback drives state
    updates  z_{t+1} = U(z_t, r_t).
    """

    # ---- task understanding (refined over time) ----
    task_understanding: str = field(
        default="",
        metadata={"description": "Current structured understanding of the task."},
    )

    # ---- hypotheses ----
    hypotheses: List[str] = field(
        default_factory=list,
        metadata={"description": "Active working hypotheses about promising approaches."},
    )

    # ---- learned patterns ----
    learned_patterns: List[str] = field(
        default_factory=list,
        metadata={"description": "Patterns discovered from execution feedback."},
    )

    # ---- attempt history (compressed) ----
    attempt_summaries: List[AttemptSummary] = field(
        default_factory=list,
        metadata={"description": "Compressed history of attempts and outcomes."},
    )

    # ---- search direction preferences ----
    preferred_directions: List[str] = field(
        default_factory=list,
        metadata={"description": "Directions considered promising."},
    )
    avoided_directions: List[str] = field(
        default_factory=list,
        metadata={"description": "Directions to avoid (dead ends)."},
    )

    # ---- confidence ----
    confidence: float = field(
        default=0.0,
        metadata={"description": "Agent confidence in its current best approach (0-1)."},
    )

    # ---- environment context (populated by recon phase) ----
    environment_context: str = field(
        default="",
        metadata={
            "description": (
                "Structured summary of the execution environment discovered "
                "during the reconnaissance phase: package versions, hardware, "
                "API compatibility notes, and known pitfalls."
            ),
        },
    )

    # ---- bookkeeping ----
    evolution_step: int = 0
    ctime: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for checkpointing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveState":
        """Deserialise from checkpoint dict."""
        if not d:
            return cls()
        # Handle attempt_summaries stored as plain dicts
        summaries = d.get("attempt_summaries", [])
        if summaries and isinstance(summaries[0], dict):
            d = dict(d)
            d["attempt_summaries"] = [
                AttemptSummary(**s) if isinstance(s, dict) else s for s in summaries
            ]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def clone(self) -> "CognitiveState":
        """Deep copy for branching (MCTS expansion)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Intrinsic quality assessment
    # ------------------------------------------------------------------

    def intrinsic_quality(self) -> float:
        """Compute an intrinsic quality score Q_intrinsic(z_t) in [0, 1].

        Measures how rich and well-calibrated this cognitive state is,
        independent of the code it produces.  Used in UCT to prefer
        states with better-formed understanding.
        """
        scores: List[float] = []

        # 1. Hypothesis richness (more diverse hypotheses → larger search space)
        scores.append(min(len(self.hypotheses) / 5.0, 1.0))

        # 2. Learned patterns (accumulated knowledge)
        scores.append(min(len(self.learned_patterns) / 5.0, 1.0))

        # 3. Direction clarity (has preferred directions AND knows dead ends)
        d_score = min(len(self.preferred_directions) / 3.0, 1.0)
        a_score = min(len(self.avoided_directions) / 3.0, 1.0)
        scores.append((d_score + a_score) / 2.0)

        # 4. Confidence calibration (does confidence match actual success rate?)
        if self.attempt_summaries:
            total = len(self.attempt_summaries)
            successes = sum(1 for a in self.attempt_summaries if not a.is_buggy)
            actual_rate = successes / total
            calibration = 1.0 - abs(self.confidence - actual_rate)
            scores.append(max(0.0, calibration))

        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def to_prompt_str(
        self,
        max_attempts: int = 15,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the cognitive state as a structured string for injection
        into operator prompts.

        Parameters
        ----------
        max_attempts : int
            Cap on how many recent attempts to include (avoids prompt bloat).
        focus_direction : str, optional
            If set, this direction is highlighted as a MANDATORY constraint
            for the current expansion child (diversity mechanism).
        cross_branch_insights : dict, optional
            Knowledge harvested from other branches for cross-pollination.
        """
        # Even at step 0, render environment_context if available
        if self.evolution_step == 0 and not self.environment_context:
            return ""  # first step and no recon data – nothing to show

        sections: List[str] = []
        sections.append("# Agent Cognitive State (accumulated understanding)\n")

        if self.environment_context:
            sections.append(
                f"## Environment Reconnaissance\n"
                f"The following was discovered by probing the execution environment. "
                f"Use this to avoid API/version incompatibilities.\n"
                f"{self.environment_context}\n"
            )

        if self.task_understanding:
            sections.append(f"## Task Understanding\n{self.task_understanding}\n")

        if self.hypotheses:
            items = "\n".join(f"- {h}" for h in self.hypotheses)
            sections.append(f"## Active Hypotheses\n{items}\n")

        if self.learned_patterns:
            items = "\n".join(f"- {p}" for p in self.learned_patterns)
            sections.append(f"## Learned Patterns\n{items}\n")

        # Focus direction: overrides general preferred_directions
        if focus_direction:
            sections.append(
                f"## **MANDATORY FOCUS DIRECTION**\n"
                f"You MUST implement the following specific approach:\n"
                f"→ {focus_direction}\n"
                f"Do NOT deviate from this direction. Build your entire solution around it.\n"
            )
        elif self.preferred_directions:
            items = "\n".join(f"- {d}" for d in self.preferred_directions)
            sections.append(f"## Promising Directions\n{items}\n")

        if self.avoided_directions:
            items = "\n".join(f"- {d}" for d in self.avoided_directions)
            sections.append(f"## Directions to Avoid\n{items}\n")

        # Cross-branch insights
        if cross_branch_insights:
            cb_sections: List[str] = []
            patterns = cross_branch_insights.get("cross_branch_patterns", [])
            if patterns:
                cb_sections.append("Patterns discovered by other exploration branches:")
                for p in patterns[:8]:
                    cb_sections.append(f"  - {p}")
            dead_ends = cross_branch_insights.get("confirmed_dead_ends", [])
            if dead_ends:
                cb_sections.append("Confirmed dead ends from other branches:")
                for d in dead_ends[:5]:
                    cb_sections.append(f"  - {d}")
            if cb_sections:
                sections.append("## Cross-Branch Insights\n" + "\n".join(cb_sections) + "\n")

        if self.attempt_summaries:
            recent = self.attempt_summaries[-max_attempts:]
            rows: List[str] = []
            for a in recent:
                metric_str = f"{a.metric:.4f}" if a.metric is not None else "buggy"
                rows.append(f"  step {a.step}: [{metric_str}] {a.approach} → {a.key_insight}")
            sections.append("## Recent Attempts\n" + "\n".join(rows) + "\n")

        sections.append(f"Confidence in current direction: {self.confidence:.2f}")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_attempt(
        self,
        step: int,
        approach: str,
        metric: Optional[float],
        is_buggy: bool,
        key_insight: str,
        max_history: int = 50,
    ) -> None:
        """Append a compressed attempt record, trimming old entries."""
        self.attempt_summaries.append(
            AttemptSummary(
                step=step,
                approach=approach,
                metric=metric,
                is_buggy=is_buggy,
                key_insight=key_insight,
            )
        )
        if len(self.attempt_summaries) > max_history:
            self.attempt_summaries = self.attempt_summaries[-max_history:]


# ======================================================================
# Multi-dimensional feedback r_t
# ======================================================================

@dataclass
class Feedback:
    """Structured multi-dimensional feedback signal r_t.

    This replaces the single (metric, is_buggy) pair with a rich signal
    that drives cognitive state evolution U(z_t, r_t) → z_{t+1}.

    Dimensions:
        metric:           raw score from evaluation (None if buggy)
        is_buggy:         whether code executed without valid output
        error_category:   classified error type (environment / logic / data / resource)
        error_pattern:    specific error pattern for learned_patterns
        trend:            metric trajectory (improving / stagnating / degrading / first)
        trend_delta:      magnitude of change vs previous best
        novelty:          how different this approach is from previous attempts (0-1)
        free_energy_drop: estimated reduction in cognitive uncertainty (0-1)
    """

    # --- core signals ---
    metric: Optional[float] = None
    is_buggy: bool = True

    # --- error classification ---
    error_category: str = ""  # "environment" | "logic" | "data" | "resource" | "none"
    error_pattern: str = ""   # e.g. "ReduceLROnPlateau(verbose) deprecated"

    # --- trend analysis ---
    trend: str = "first"      # "improving" | "stagnating" | "degrading" | "first"
    trend_delta: float = 0.0  # signed change vs previous best

    # --- exploration signals ---
    novelty: float = 0.0      # 0 = identical to previous, 1 = completely new approach
    free_energy_drop: float = 0.0  # estimated uncertainty reduction

    def to_prompt_str(self) -> str:
        """Render feedback as structured text for reflect_op prompt."""
        lines = ["## Feedback Signal r_t"]

        if self.is_buggy:
            lines.append(f"**Status**: BUGGY (error_category: {self.error_category})")
            if self.error_pattern:
                lines.append(f"**Error pattern**: {self.error_pattern}")
        else:
            lines.append(f"**Status**: SUCCESS (metric: {self.metric})")

        lines.append(f"**Trend**: {self.trend} (delta: {self.trend_delta:+.4f})")
        lines.append(f"**Novelty**: {self.novelty:.2f}")
        lines.append(f"**Free energy drop**: {self.free_energy_drop:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Error classification patterns
_ERROR_PATTERNS = {
    "environment": [
        ("ReduceLROnPlateau", "verbose", "ReduceLROnPlateau(verbose) removed in PyTorch>=2.4"),
        ("predict_with_generate", "", "predict_with_generate removed in transformers>=4.46"),
        ("nvrtc", "libnvrtc-builtins", "CUDA nvrtc-builtins.so version mismatch"),
        ("No module named", "", "missing module"),
        ("ModuleNotFoundError", "", "missing module"),
        ("subprocesses has abruptly died", "", "datasets.map(num_proc) multiprocess crash"),
        ("HF_HUB_OFFLINE", "", "HuggingFace offline mode blocking download"),
    ],
    "resource": [
        ("OutOfMemoryError", "", "CUDA OOM"),
        ("CUDA out of memory", "", "CUDA OOM"),
        ("Killed", "", "process killed (likely OOM)"),
        ("TimeoutError", "", "execution timeout"),
    ],
    "data": [
        ("KeyError", "", "missing column/key in data"),
        ("FileNotFoundError", "", "missing file"),
        ("ValueError: setting an array element", "", "inhomogeneous array shape"),
    ],
}


def classify_error(term_out: str) -> tuple:
    """Classify an error from terminal output into (category, pattern).

    Returns ("none", "") if no error detected.
    """
    for category, patterns in _ERROR_PATTERNS.items():
        for keywords in patterns:
            key1, key2, label = keywords
            if key1 in term_out and (not key2 or key2 in term_out):
                return category, label
    # Generic logic error
    if "Error" in term_out or "Exception" in term_out:
        return "logic", ""
    return "none", ""


def compute_novelty(code: str, previous_codes: List[str], max_compare: int = 5) -> float:
    """Estimate novelty of code vs previous attempts (0=identical, 1=completely new).

    Uses a simple set-of-lines Jaccard distance.
    """
    if not previous_codes or not code:
        return 1.0

    current_lines = set(line.strip() for line in code.split("\n") if line.strip())
    if not current_lines:
        return 1.0

    similarities = []
    for prev in previous_codes[-max_compare:]:
        prev_lines = set(line.strip() for line in prev.split("\n") if line.strip())
        if not prev_lines:
            continue
        intersection = len(current_lines & prev_lines)
        union = len(current_lines | prev_lines)
        similarities.append(intersection / union if union > 0 else 0.0)

    if not similarities:
        return 1.0

    max_sim = max(similarities)
    return 1.0 - max_sim


def build_feedback(
    node: "Node",
    journal: "Journal",
    lower_is_better: bool,
) -> "Feedback":
    """Construct multi-dimensional feedback r_t from a completed node.

    This is the core function that transforms raw execution results
    into the structured signal that drives cognitive state evolution.
    """
    # --- Error classification ---
    term_out = node.term_out if node.term_out else ""
    error_category, error_pattern = classify_error(term_out)
    if not node.is_buggy:
        error_category = "none"
        error_pattern = ""

    # --- Metric and trend ---
    metric_val = None
    if hasattr(node.metric, "value") and node.metric.value is not None:
        metric_val = node.metric.value

    # Find previous best for trend
    prev_metrics = []
    for n in journal.nodes:
        if n is node or n.is_buggy:
            continue
        if hasattr(n.metric, "value") and n.metric.value is not None:
            prev_metrics.append(n.metric.value)

    if metric_val is not None and prev_metrics:
        prev_best = min(prev_metrics) if lower_is_better else max(prev_metrics)
        trend_delta = prev_best - metric_val if lower_is_better else metric_val - prev_best
        # trend_delta > 0 means improvement
        if abs(trend_delta) < 1e-6:
            trend = "stagnating"
        elif trend_delta > 0:
            trend = "improving"
        else:
            trend = "degrading"
    elif metric_val is not None:
        trend = "first"
        trend_delta = 0.0
    else:
        trend = "first" if not prev_metrics else "degrading"
        trend_delta = 0.0

    # --- Novelty ---
    previous_codes = [n.code for n in journal.nodes if n.code and n is not node]
    novelty = compute_novelty(node.code or "", previous_codes)

    # --- Free energy drop estimate ---
    # Proxy: higher novelty + not buggy → more information gained → bigger drop
    if not node.is_buggy:
        free_energy_drop = 0.3 + 0.4 * novelty + (0.3 if trend == "improving" else 0.0)
    elif error_pattern:
        # Buggy but identified pattern → some info gained
        free_energy_drop = 0.2 + 0.2 * novelty
    else:
        # Buggy with no clear pattern → minimal info
        free_energy_drop = 0.1 * novelty

    return Feedback(
        metric=metric_val,
        is_buggy=node.is_buggy,
        error_category=error_category,
        error_pattern=error_pattern,
        trend=trend,
        trend_delta=trend_delta,
        novelty=novelty,
        free_energy_drop=min(free_energy_drop, 1.0),
    )
