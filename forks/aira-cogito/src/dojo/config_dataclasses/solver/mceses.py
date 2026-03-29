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
            assert self.num_children >= 1, "num_children must be >= 1 for tree search"
            assert self.uct_c >= 0, "uct_c must be non-negative"
            w_sum = self.backprop_metric_weight + self.backprop_validity_weight + self.backprop_improvement_weight
            assert abs(w_sum - 1.0) < 0.01, f"backprop weights must sum to 1.0, got {w_sum}"
        assert self.intervention_mode in ("natural", "ablated", "scrambled", "frozen"), (
            f"Unknown intervention_mode: {self.intervention_mode}"
        )
