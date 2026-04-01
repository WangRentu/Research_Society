# Copyright (c) 2026 Research Society
# MC-ESES (Monte Carlo Endogenous State Evolution Search) solver configuration dataclass.

from dataclasses import dataclass, field

from dojo.config_dataclasses.solver.base import SolverConfig


@dataclass
class MCESESSolverConfig(SolverConfig):
    """Configuration for the MC-ESES (Monte Carlo Endogenous State Evolution Search) solver."""

    # --- Search Configuration (inherited from Greedy-like behaviour) ---
    improvement_steps: int = field(
        default=5,
        metadata={"description": "Number of improvement iterations to perform."},
    )

    data_preview: bool = field(
        default=True,
        metadata={"description": "Whether to provide the agent with a data preview."},
    )

    # --- Debugging ---
    max_debug_depth: int = field(
        default=3,
        metadata={"description": "Maximum depth of consecutive debug steps."},
    )

    debug_prob: float = field(
        default=0.5,
        metadata={"description": "Probability of debugging a buggy node vs drafting/improving."},
    )

    # --- Drafting ---
    num_drafts: int = field(
        default=5,
        metadata={"description": "Number of initial drafts before switching to improve."},
    )

    # --- MC-ESES-specific ---
    reflect_after_every_step: bool = field(
        default=True,
        metadata={"description": "Run reflect_op after every step to update cognitive state."},
    )

    managed_evolution: bool = field(
        default=True,
        metadata={"description": "Enable managed cognitive-state evolution heuristics."},
    )

    triggered_reflection: bool = field(
        default=True,
        metadata={"description": "Only call reflect_op on informative events."},
    )

    reflect_on_routine: bool = field(
        default=False,
        metadata={"description": "Also reflect on routine steps when triggered_reflection is enabled."},
    )

    max_state_history: int = field(
        default=50,
        metadata={"description": "Max attempt summaries to keep in cognitive state."},
    )

    # --- Phase B: MC-ESES (MCTS over cognitive state space) ---
    use_tree_search: bool = field(
        default=False,
        metadata={"description": "Enable MCTS over cognitive state space (Phase B / MC-ESES)."},
    )

    uct_c: float = field(
        default=0.25,
        metadata={"description": "UCT exploration constant for cognitive state tree."},
    )

    num_children: int = field(
        default=5,
        metadata={"description": "Number of children per leaf expansion in the CS tree."},
    )

    max_cs_tree_depth: int = field(
        default=10,
        metadata={"description": "Maximum depth of the cognitive state tree."},
    )

    max_debug_time: float = field(
        default=600.0,
        metadata={"description": "Maximum time (seconds) for debug cycles during CS expansion."},
    )

    # --- Phase B: strategy-driven action selection ---
    strategy_driven: bool = field(
        default=True,
        metadata={"description": "Use z_t confidence to select draft vs improve action."},
    )

    # --- Phase B: expansion diversity ---
    diverse_expansion: bool = field(
        default=True,
        metadata={"description": "Assign different preferred_directions to each expansion child."},
    )

    adaptive_children: bool = field(
        default=True,
        metadata={"description": "Adapt branching factor to branch confidence."},
    )
    low_conf_children: int = field(
        default=5,
        metadata={"description": "Children to expand when confidence is low."},
    )
    mid_conf_children: int = field(
        default=3,
        metadata={"description": "Children to expand when confidence is medium."},
    )
    high_conf_children: int = field(
        default=1,
        metadata={"description": "Children to expand when confidence is high."},
    )
    low_conf_threshold: float = field(
        default=0.3,
        metadata={"description": "Confidence threshold below which the search stays exploratory."},
    )
    high_conf_threshold: float = field(
        default=0.7,
        metadata={"description": "Confidence threshold above which branching becomes conservative."},
    )

    state_guided_policy: bool = field(
        default=True,
        metadata={"description": "Let branch-local cognitive state influence draft vs improve decisions."},
    )
    avoid_dead_end_improve: bool = field(
        default=True,
        metadata={"description": "Avoid improving a branch that is already marked as a dead end."},
    )

    plateau_window: int = field(
        default=3,
        metadata={"description": "Window size for plateau detection and managed decay."},
    )
    bug_streak_window: int = field(
        default=3,
        metadata={"description": "Consecutive buggy attempts needed to trigger a hard reset."},
    )
    score_jump_threshold: float = field(
        default=0.05,
        metadata={"description": "Relative metric change threshold for score-jump reflection triggers."},
    )
    state_decay_on_plateau: bool = field(
        default=True,
        metadata={"description": "Apply confidence decay after plateau events."},
    )
    plateau_decay_rate: float = field(
        default=0.3,
        metadata={"description": "Fraction of confidence removed on plateau."},
    )
    hard_reset_on_bug_streak: bool = field(
        default=True,
        metadata={"description": "Apply a stronger state reset after repeated buggy attempts."},
    )

    # --- Phase B: multi-dimensional backpropagation ---
    backprop_metric_weight: float = field(
        default=0.6,
        metadata={"description": "Weight for metric signal in composite backprop value."},
    )
    backprop_validity_weight: float = field(
        default=0.2,
        metadata={"description": "Weight for validity rate in composite backprop value."},
    )
    backprop_improvement_weight: float = field(
        default=0.2,
        metadata={"description": "Weight for metric improvement signal in composite backprop value."},
    )

    # --- Phase B: intrinsic quality in UCT ---
    intrinsic_quality_weight: float = field(
        default=0.2,
        metadata={"description": "Weight of intrinsic state quality in UCT value (0 = disabled)."},
    )

    # --- Phase B: cross-branch knowledge transfer ---
    crossover_enabled: bool = field(
        default=True,
        metadata={"description": "Enable cross-branch cognitive state knowledge transfer."},
    )
    crossover_top_k: int = field(
        default=3,
        metadata={"description": "Number of top leaves to harvest insights from for crossover."},
    )

    # --- Experiment instrumentation ---
    save_trajectory: bool = field(
        default=True,
        metadata={"description": "Save z_t snapshots to trajectory.jsonl for analysis."},
    )

    # --- Causal intervention (Exp 3) ---
    intervention_mode: str = field(
        default="natural",
        metadata={
            "description": (
                "Causal intervention on z_t. "
                "'natural': no intervention (control). "
                "'ablated': reset z_t to z_0 after every reflect (no evolution). "
                "'scrambled': replace z_t with state loaded from scramble_source_path. "
                "'frozen': freeze z_t at intervention_freeze_step (no further updates)."
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
        if self.use_tree_search:
            assert self.num_children >= 1, "num_children must be >= 1"
            assert self.uct_c >= 0, "uct_c must be non-negative"
            w_sum = self.backprop_metric_weight + self.backprop_validity_weight + self.backprop_improvement_weight
            assert abs(w_sum - 1.0) < 0.01, f"backprop weights must sum to 1.0, got {w_sum}"
        assert self.low_conf_children >= 1 and self.mid_conf_children >= 1 and self.high_conf_children >= 1, (
            "adaptive child counts must all be >= 1"
        )
        assert 0.0 <= self.low_conf_threshold <= self.high_conf_threshold <= 1.0, (
            "confidence thresholds must satisfy 0 <= low_conf_threshold <= high_conf_threshold <= 1"
        )
        assert 0.0 <= self.plateau_decay_rate <= 1.0, "plateau_decay_rate must be in [0, 1]"
        assert self.intervention_mode in ("natural", "ablated", "scrambled", "frozen"), (
            f"Unknown intervention_mode: {self.intervention_mode}"
        )
