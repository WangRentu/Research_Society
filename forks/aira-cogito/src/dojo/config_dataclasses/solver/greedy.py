# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from dojo.config_dataclasses.solver.base import SolverConfig


@dataclass
class GreedySolverConfig(SolverConfig):
    # --- Search Configuration ---
    improvement_steps: int = field(
        default=5,
        metadata={
            "description": "Number of improvement iterations to perform.",
            "example": 3,
        },
    )

    data_preview: bool = field(
        default=False,
        metadata={
            "description": "Whether to provide the agent with a preview of the data before execution.",
            "example": True,
        },
    )

    # --- Debugging Configuration ---
    max_debug_depth: int = field(
        default=3,
        metadata={
            "description": "Maximum depth of debugging analysis.",
            "example": 2,
        },
    )

    debug_prob: float = field(
        default=0.5,
        metadata={
            "description": "Probability of running a debug step in the process.",
            "example": 0.3,
        },
    )

    # --- Drafting Configuration ---
    num_drafts: int = field(
        default=5,
        metadata={
            "description": "Number of draft outputs to generate for selection.",
            "example": 3,
        },
    )

    # --- Cognitive State Evolution ---
    use_cognitive_state: bool = field(
        default=False,
        metadata={
            "description": (
                "Enable CognitiveState z_t + reflect_op. "
                "When True, the Greedy search structure is unchanged "
                "(5 drafts + improve chains) but each step updates z_t "
                "and injects it into operator prompts."
            ),
        },
    )
    reflect_after_every_step: bool = field(
        default=True,
        metadata={"description": "Call reflect_op after every step (vs only after score changes)."},
    )

    # --- Managed evolution (NAT+) ---
    managed_evolution: bool = field(
        default=False,
        metadata={
            "description": (
                "Enable managed evolution for Greedy (NAT+). "
                "Adds triggered reflection, lightweight updates, and state decay."
            ),
        },
    )
    triggered_reflection: bool = field(
        default=True,
        metadata={"description": "Only reflect on meaningful events (new_error, score_jump, bug_streak, plateau)."},
    )
    reflect_on_routine: bool = field(
        default=False,
        metadata={"description": "Whether to LLM-reflect on routine (non-triggered) steps."},
    )
    plateau_window: int = field(
        default=3,
        metadata={"description": "Number of consecutive non-improving valid steps to detect plateau."},
    )
    bug_streak_window: int = field(
        default=3,
        metadata={"description": "Number of consecutive buggy steps to trigger hard reset."},
    )
    score_jump_threshold: float = field(
        default=0.05,
        metadata={"description": "Relative change in metric to classify as score_jump."},
    )
    state_decay_on_plateau: bool = field(
        default=True,
        metadata={"description": "Decay confidence when plateau detected."},
    )
    plateau_decay_rate: float = field(
        default=0.3,
        metadata={"description": "Multiplicative decay rate for confidence on plateau."},
    )
    hard_reset_on_bug_streak: bool = field(
        default=True,
        metadata={"description": "Hard reset confidence/hypotheses on long bug streaks."},
    )
    max_state_history: int = field(
        default=50,
        metadata={"description": "Max attempt summaries to keep in cognitive state."},
    )

    # --- Experiment instrumentation ---
    save_trajectory: bool = field(
        default=False,
        metadata={"description": "Save z_t snapshots to trajectory.jsonl for analysis."},
    )
    intervention_mode: str = field(
        default="natural",
        metadata={
            "description": (
                "Causal intervention on z_t. "
                "'natural': no intervention. "
                "'ablated': reset z_t to z_0 after every reflect. "
                "'scrambled': replace z_t with state from scramble_source_path. "
                "'frozen': freeze z_t at intervention_freeze_step."
            ),
        },
    )
    intervention_freeze_step: int = field(
        default=5,
        metadata={"description": "Step at which to freeze z_t (frozen mode only)."},
    )
    scramble_source_path: str = field(
        default="",
        metadata={"description": "Path to cognitive_state.json from a different task (scrambled mode)."},
    )

    def validate(self) -> None:
        super().validate()
        if self.use_cognitive_state:
            assert self.intervention_mode in ("natural", "ablated", "scrambled", "frozen"), (
                f"Unknown intervention_mode: {self.intervention_mode}"
            )
