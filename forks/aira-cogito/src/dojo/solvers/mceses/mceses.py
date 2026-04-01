# Copyright (c) 2026 Research Society
# MC-ESES: Monte Carlo Endogenous State Evolution Search
#
# Phase A: Greedy-like solver augmented with a persistent CognitiveState
# and a reflect_op that updates the state after each step.
#
# Phase B (MC-ESES): Monte Carlo tree search over COGNITIVE STATE space.
# Instead of searching over code solutions, the UCT tree consists of
# CognitiveStateNodes.  Code is generated as a projection E(z_t),
# executed, and the result drives state evolution U(z_t, r_t) → z_{t+1}.
#
# Activated via  use_tree_search: true  in MCESESSolverConfig.

import json
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dojo.core.solvers.base import Solver
from dojo.core.solvers.operators.analyze import analyze_op
from dojo.core.solvers.operators.core import execute_op_plan_code
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.operators.reflect import reflect_op
from dojo.core.solvers.utils import data_preview
from dojo.core.solvers.utils.cognitive_state import CognitiveState, Feedback, build_feedback
from dojo.core.solvers.utils.cognitive_state_tree import (
    CognitiveStateNode,
    deserialize_cs_tree,
    serialize_cs_tree,
    tree_stats,
    uct_value,
)
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.core.solvers.utils.search_exporter import export_search_results
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
from dojo.solvers.mceses.instrumentation import (
    TriggerType,
    classify_trigger,
    compute_state_delta,
    save_trajectory_snapshot,
)
from dojo.solvers.utils import get_complextiy_level
from dojo.utils.code_parsing import parse_json_output
from dojo.utils.environment import parse_pip_list_output
from dojo.utils.state import MCESESState


class MCESES(Solver):
    """Monte Carlo Endogenous State Evolution Search.

    Phase A (use_tree_search=False):
        Greedy search with persistent CognitiveState and reflect_op.

    Phase B / MC-ESES (use_tree_search=True):
        Monte Carlo tree search over cognitive state space.
        UCT selects which CognitiveState to expand; expansion generates
        code via E(z_t), executes it, and reflects to produce child states.
    """

    def __init__(self, cfg, task_info):
        super().__init__(cfg, task_info=task_info)
        self.journal = Journal()
        self.data_preview_str: Optional[str] = None
        self.cognitive_state = CognitiveState()

        self.task_desc = task_info[TASK_DESCRIPTION]
        self.lower_is_better = task_info.get("lower_is_better", None)
        assert self.lower_is_better is not None

        self.setup_operators()
        self.state = MCESESState()

        # Phase B: cognitive state tree (initialised in _run_tree_search)
        self.cs_root: Optional[CognitiveStateNode] = None

        # --- Experiment instrumentation ---
        self._trajectory_path: Optional[Path] = None
        self._prev_error_categories: List[str] = []
        self._recent_metrics: List[Optional[float]] = []
        self._has_any_valid: bool = False
        self._frozen_state: Optional[CognitiveState] = None
        self._scrambled_state: Optional[CognitiveState] = None

        # Load scrambled state if configured
        if self.cfg.intervention_mode == "scrambled" and self.cfg.scramble_source_path:
            src = Path(self.cfg.scramble_source_path)
            if src.exists():
                with open(src) as f:
                    self._scrambled_state = CognitiveState.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Operator setup
    # ------------------------------------------------------------------

    def setup_operators(self):
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])
        reflect_llm = GenericLLM(self.cfg.operators["reflect"])

        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        self.draft_fn = partial(draft_op, draft_llm, self.cfg, self.memory_op)
        self.improve_fn = partial(improve_op, improve_llm, self.cfg, self.memory_op)
        self.debug_fn = partial(debug_op, debug_llm, self.cfg, self.debug_memory_op)
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)
        self.reflect_fn = partial(reflect_op, reflect_llm, self.cfg)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self):
        super().save_checkpoint()

        # Save journal
        journal_sd = self.journal.node_list()
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        with open(journal_path, "w") as f:
            for node in journal_sd:
                f.write(json.dumps(node) + "\n")

        # Save cognitive state
        cs_path = Path(self.cfg.checkpoint_path) / "cognitive_state.json"
        with open(cs_path, "w") as f:
            json.dump(self.cognitive_state.to_dict(), f, indent=2)

        # Phase B: save cognitive state tree
        if self.cfg.use_tree_search and self.cs_root is not None:
            cs_tree_path = Path(self.cfg.checkpoint_path) / "cs_tree.json"
            with open(cs_tree_path, "w") as f:
                json.dump(serialize_cs_tree(self.cs_root), f)

        self.logger.info(f"MC-ESES checkpoint saved to {self.cfg.checkpoint_path}")

    def load_checkpoint(self):
        super().load_checkpoint()

        # Load journal
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        if journal_path.exists():
            self.logger.info(f"Loading journal from {journal_path}")
            with open(journal_path, "r") as f:
                journal_export = [json.loads(line) for line in f]
            self.journal = Journal.from_export_data({"nodes": journal_export})

        # Load cognitive state
        cs_path = Path(self.cfg.checkpoint_path) / "cognitive_state.json"
        if cs_path.exists():
            self.logger.info(f"Loading cognitive state from {cs_path}")
            with open(cs_path, "r") as f:
                cs_dict = json.load(f)
            self.cognitive_state = CognitiveState.from_dict(cs_dict)

        # Phase B: load cognitive state tree
        if self.cfg.use_tree_search:
            cs_tree_path = Path(self.cfg.checkpoint_path) / "cs_tree.json"
            if cs_tree_path.exists():
                self.logger.info(f"Loading CS tree from {cs_tree_path}")
                with open(cs_tree_path, "r") as f:
                    self.cs_root = deserialize_cs_tree(json.load(f))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create_root_node(self):
        """Create an empty root node for Journal compatibility."""
        self.root_node = Node(
            code="",
            plan="",
            analysis="",
            metric=WorstMetricValue(maximize=not self.lower_is_better),
            is_buggy=True,
        )
        self.root_node.absorb_exec_result(None)
        self.journal.append(self.root_node)
        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": 0},
            "JOURNAL",
            step=self.state.current_step,
        )
        self.state.current_step += 1

    # ------------------------------------------------------------------
    # Main loop — z_t is the protagonist, Node is a byproduct
    # ------------------------------------------------------------------

    def __call__(self, task, state):
        """MC-ESES entry point.

        K=1 is treated as a degenerate single-trajectory chain: the solver
        still operates over cognitive-state nodes, but each expansion creates
        exactly one child. K>1 yields the full branching tree-search mode.
        """
        if self.cfg.num_children <= 1:
            return self._run_chain_search(task, state)
        return self._run_tree_search(task, state)

    def _run_chain_search(self, task, state):
        """Run the degenerate K=1 MC-ESES mode.

        This reuses the same cognitive-state tree machinery as the branching
        solver, but with one child per expansion. The resulting search is a
        single trajectory over branch-local cognitive states, which keeps the
        implementation aligned with K>1 while avoiding a separate code path.
        """
        self.logger.info("Starting MC-ESES chain search (degenerate K=1 mode)")
        return self._run_tree_search(task, state)

    # ------------------------------------------------------------------
    # Experiment instrumentation: intervention + trajectory
    # ------------------------------------------------------------------

    def _init_trajectory(self) -> None:
        """Initialise trajectory file path."""
        if self.cfg.save_trajectory and self._trajectory_path is None:
            self._trajectory_path = Path(self.cfg.checkpoint_path) / "trajectory.jsonl"

    def _apply_intervention(self, z_after_reflect: CognitiveState, step: int) -> CognitiveState:
        """Apply causal intervention to z_t after reflect (Exp 3).

        Returns the (possibly modified) cognitive state.
        """
        mode = self.cfg.intervention_mode
        if mode == "natural":
            return z_after_reflect

        if mode == "ablated":
            # Reset task-specific knowledge but preserve neutral search behaviour.
            # - environment_context: kept (task-agnostic setup info)
            # - evolution_step: kept (so search_policy exits bootstrap)
            # - confidence: set to 0.5 (neutral — allows improve/debug, not only draft)
            # - attempt_summaries: kept (search_policy uses them for debug detection)
            # This ensures the same draft/improve/debug action mix as natural,
            # only without hypotheses, learned_patterns, or preferred_directions.
            blank = CognitiveState()
            blank.environment_context = z_after_reflect.environment_context
            blank.evolution_step = z_after_reflect.evolution_step
            blank.confidence = 0.5
            blank.attempt_summaries = z_after_reflect.attempt_summaries
            return blank

        if mode == "scrambled":
            if self._scrambled_state is not None:
                s = self._scrambled_state.clone()
                # Preserve environment_context and search-neutral fields
                s.environment_context = z_after_reflect.environment_context
                s.evolution_step = z_after_reflect.evolution_step
                s.attempt_summaries = z_after_reflect.attempt_summaries
                return s
            # Fallback: behave as ablated if no scramble source
            blank = CognitiveState()
            blank.environment_context = z_after_reflect.environment_context
            blank.evolution_step = z_after_reflect.evolution_step
            blank.confidence = 0.5
            blank.attempt_summaries = z_after_reflect.attempt_summaries
            return blank

        if mode == "frozen":
            if step <= self.cfg.intervention_freeze_step:
                # Before freeze point: allow evolution, but snapshot at freeze point
                if step == self.cfg.intervention_freeze_step:
                    self._frozen_state = z_after_reflect.clone()
                return z_after_reflect
            else:
                # After freeze point: always return the frozen snapshot
                if self._frozen_state is not None:
                    return self._frozen_state.clone()
                return z_after_reflect

        return z_after_reflect

    def _record_step(
        self,
        step: int,
        z_before: CognitiveState,
        z_after: CognitiveState,
        feedback: Feedback,
        action: str,
        code_node: Node,
    ) -> None:
        """Record trajectory snapshot and update tracking state."""
        if not self.cfg.save_trajectory or self._trajectory_path is None:
            return

        trigger = classify_trigger(
            feedback=feedback,
            step=z_before.evolution_step,
            prev_error_categories=self._prev_error_categories,
            recent_metrics=self._recent_metrics[-3:],
            has_any_valid=self._has_any_valid,
        )

        delta = compute_state_delta(z_before, z_after)

        metric_val = None
        if not code_node.is_buggy and hasattr(code_node.metric, "value"):
            metric_val = code_node.metric.value

        save_trajectory_snapshot(
            path=self._trajectory_path,
            step=step,
            z_before=z_before,
            z_after=z_after,
            feedback=feedback,
            action=action,
            trigger_type=trigger,
            state_delta=delta,
            metric_value=metric_val,
            intervention_mode=self.cfg.intervention_mode,
        )

        # Update tracking state for next trigger classification
        if feedback.error_category and feedback.error_category != "none":
            if feedback.error_category not in self._prev_error_categories:
                self._prev_error_categories.append(feedback.error_category)
        if feedback.metric is not None:
            self._recent_metrics.append(feedback.metric)
        if not feedback.is_buggy:
            self._has_any_valid = True

    # ------------------------------------------------------------------
    # z_t-driven search policy — cognitive state drives all decisions
    # ------------------------------------------------------------------

    def search_policy(self) -> Tuple[str, Optional[Node], Optional[str]]:
        """Let z_t decide what action to take.

        Returns (action, parent_node, focus_direction) where:
            action:          "draft" | "improve" | "debug"
            parent_node:     Node to improve/debug (None for draft)
            focus_direction: specific direction from z_t.preferred_directions (or None)

        Decision logic (endogenous — driven by z_t, not hardcoded):

        1. Bootstrap phase (evolution_step == 0): always draft to get initial data.
        2. All recent attempts buggy + z_t has learned_patterns about the bug:
           → debug (z_t knows what's wrong, try to fix it)
        3. z_t.confidence < 0.3 or z_t has untried preferred_directions:
           → draft a new approach (explore — z_t is uncertain or has new ideas)
        4. z_t.confidence >= 0.3 and good nodes exist:
           → improve best node (exploit — z_t believes current direction is promising)
        5. Fallback: draft (explore)
        """
        z = self.cognitive_state
        step = self.state.current_step

        # --- Bootstrap: first few steps always draft ---
        if z.evolution_step == 0 or len(self.journal.draft_nodes) < 2:
            self.logger.info(
                f"z_t Policy: DRAFT (bootstrap, evolution_step={z.evolution_step})"
            )
            return "draft", None, None

        # --- Gather state signals ---
        best_node = self.journal.get_best_node()
        has_good_nodes = best_node is not None and not best_node.is_buggy

        # Check if recent attempts are all buggy (last N steps)
        recent_window = min(3, len(self.journal.nodes) - 1)  # exclude root
        recent_nodes = [n for n in self.journal.nodes[-recent_window:] if n.code]
        recent_all_buggy = all(n.is_buggy for n in recent_nodes) if recent_nodes else False

        # Debuggable nodes
        debuggable = [
            n for n in self.journal.buggy_nodes
            if n.is_leaf and n.debug_depth <= self.cfg.max_debug_depth
        ]

        # Directions already tried (from attempt_summaries)
        tried_approaches = set()
        for a in z.attempt_summaries:
            tried_approaches.add(a.approach[:50])  # rough dedup

        # Untried preferred directions
        untried_directions = [
            d for d in z.preferred_directions
            if not any(d[:30] in t for t in tried_approaches)
        ]

        # --- Decision logic ---

        # 2. Recent attempts all buggy → debug
        #    With learned_patterns: z_t guides the fix (NAT advantage).
        #    Without learned_patterns: still debug — same action, just less
        #    informed prompt content (fair ABL baseline behaviour).
        if recent_all_buggy and debuggable:
            target = debuggable[-1]  # most recent buggy node
            self.logger.info(
                f"z_t Policy: DEBUG (recent {recent_window} steps all buggy, "
                f"z_t has {len(z.learned_patterns)} learned patterns)"
            )
            return "debug", target, None

        # 3. Low confidence or has untried ideas → draft new approach
        if z.confidence < 0.3:
            focus = untried_directions[0] if untried_directions else None
            self.logger.info(
                f"z_t Policy: DRAFT (confidence={z.confidence:.2f} < 0.3, "
                f"untried_directions={len(untried_directions)})"
            )
            return "draft", None, focus

        if untried_directions and not has_good_nodes:
            focus = untried_directions[0]
            self.logger.info(
                f"z_t Policy: DRAFT (no good nodes, trying direction: {focus[:50]})"
            )
            return "draft", None, focus

        # 4. Confident + good nodes → improve
        if has_good_nodes and z.confidence >= 0.3:
            # Pick direction from z_t.preferred_directions if available
            focus = z.preferred_directions[0] if z.preferred_directions else None
            self.logger.info(
                f"z_t Policy: IMPROVE (confidence={z.confidence:.2f}, "
                f"best_metric={best_node.metric.value})"
            )
            return "improve", best_node, focus

        # 5. Fallback: draft
        focus = untried_directions[0] if untried_directions else None
        self.logger.info(f"z_t Policy: DRAFT (fallback)")
        return "draft", None, focus

    # ------------------------------------------------------------------
    # Operators (with cognitive state injection)
    # ------------------------------------------------------------------

    def _get_cognitive_context(
        self,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
    ) -> Optional[str]:
        """Render cognitive state for prompt injection."""
        s = self.cognitive_state.to_prompt_str(
            focus_direction=focus_direction,
            cross_branch_insights=cross_branch_insights,
        )
        return s if s else None

    def _draft(
        self,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
    ) -> Node:
        self.logger.info(f"Step {self.state.current_step}: Drafting new solution (MC-ESES)")
        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview_str,
            get_complextiy_level(num=len(self.journal.draft_nodes)) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(
                focus_direction=focus_direction,
                cross_branch_insights=cross_branch_insights,
            ),
        )
        node = Node(
            plan=plan, code=code,
            operators_used=["draft"], operators_metrics=[metrics],
            parents=[self.root_node],
        )
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        return node

    def _improve(
        self,
        parent_node: Node,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
    ) -> Node:
        self.logger.info(f"Step {self.state.current_step}: Improving solution (MC-ESES)")
        plan, code, metrics = execute_op_plan_code(
            self.improve_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            get_complextiy_level(parent_node) if self.cfg.use_complexity else None,
            self.data_preview_str,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(
                focus_direction=focus_direction,
                cross_branch_insights=cross_branch_insights,
            ),
        )
        node = Node(
            plan=plan, code=code,
            parents=[parent_node],
            operators_used=["improve"], operators_metrics=[metrics],
        )
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        return node

    def _debug(self, parent_node: Node) -> Node:
        self.logger.info(f"Step {self.state.current_step}: Debugging solution (MC-ESES)")
        plan, code, metrics = execute_op_plan_code(
            self.debug_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview_str,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(),
        )
        node = Node(
            plan=plan, code=code,
            parents=[parent_node],
            operators_used=["debug"], operators_metrics=[metrics],
        )
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        return node

    def _analyze(self, node: Node) -> Union[str, dict]:
        analysis, metrics = self.analyze_fn(self.task_desc, node)
        node.operators_used.append("analysis")
        node.operators_metrics.append(metrics)
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        return analysis

    def _reflect(self, node: Node, feedback: Optional[Feedback] = None) -> None:
        """Run reflect_op to update cognitive state: z_{t+1} = U(z_t, r_t).

        If feedback is provided, it is injected into the reflect prompt so the
        LLM sees the full multi-dimensional signal, not just raw terminal output.
        """
        self.logger.info(f"Step {self.state.current_step}: Reflecting on results (MC-ESES)")

        # Inject r_t into the node's terminal output so reflect_op can see it
        if feedback is not None:
            node._term_out.append(f"\n\n{feedback.to_prompt_str()}")
            self.logger.info(
                f"r_t: trend={feedback.trend}({feedback.trend_delta:+.4f}), "
                f"novelty={feedback.novelty:.2f}, error={feedback.error_category}/{feedback.error_pattern}, "
                f"free_energy_drop={feedback.free_energy_drop:.2f}"
            )

        updated_state, metrics = self.reflect_fn(
            self.task_desc,
            self.cognitive_state,
            node,
            self.journal,
        )
        self.cognitive_state = updated_state
        node.operators_used.append("reflect")
        node.operators_metrics.append(metrics)
        self.logger.info(
            f"Cognitive state updated: evolution_step={self.cognitive_state.evolution_step}, "
            f"confidence={self.cognitive_state.confidence:.2f}, "
            f"hypotheses={len(self.cognitive_state.hypotheses)}, "
            f"patterns={len(self.cognitive_state.learned_patterns)}"
        )

    # (step() removed — logic inlined into _run_cognitive_evolution)

    # ------------------------------------------------------------------
    # Eval result parsing (same as Greedy)
    # ------------------------------------------------------------------

    def parse_eval_result(self, node: Node, eval_result: Dict[str, Any]):
        if isinstance(eval_result, dict):
            assert EXECUTION_OUTPUT in eval_result
        else:
            raise ValueError(f"Unexpected eval_result type: {type(eval_result)}")

        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])

        try:
            response = self._analyze(node)
        except Exception as e:
            self.logger.error(f"Error during analysis operator: {str(e)}")
            response = {}

        response = parse_json_output(response)
        if not isinstance(response, dict):
            response = {}

        if len(response) == 0:
            response = {"metric": None, "summary": "", "is_bug": True}
        else:
            response.setdefault("metric", None)
            response.setdefault("summary", "")
            if "is_bug" not in response:
                response["is_bug"] = response["metric"] is None

        if not isinstance(response["metric"], (float, int)):
            response["metric"] = None

        if eval_result.get(VALIDATION_FITNESS, None) is not None:
            response["metric"] = float(eval_result[VALIDATION_FITNESS])

        node.analysis = response["summary"]

        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {})
        if self.cfg.use_test_score:
            test_score = aux_eval_info.get("score", None)
            aux_eval_info["validation_score"] = response["metric"]
            response["metric"] = test_score

        valid_solution = eval_result.get(VALID_SOLUTION, True)
        validity_feedback = eval_result.get(VALID_SOLUTION_FEEDBACK, None)
        if validity_feedback is not None:
            aux_eval_info["validity_feedback"] = validity_feedback
            node._term_out.append(f"\n\n submission.csv Grader Feedback: {validity_feedback}")
        else:
            aux_eval_info["validity_feedback"] = "submission grader feedback not available"

        node.is_buggy = (
            response["is_bug"]
            or (not node.exit_code == 0)
            or (response["metric"] is None)
            or (not valid_solution)
        )

        if node.is_buggy:
            node.metric = WorstMetricValue(info=aux_eval_info)
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not self.lower_is_better, info=aux_eval_info
            )

    # ==================================================================
    # Phase B: MC-ESES — tree search over cognitive state space
    # ==================================================================

    def _run_tree_search(self, task, state):
        """Phase B main loop: MC-ESES tree search over cognitive states."""
        self.logger.info("Starting MC-ESES search (Phase B: MCTS over CognitiveState space)")

        self.create_root_node()
        self.discover_packages(state)
        self._init_trajectory()

        # Environment reconnaissance: probe the execution environment
        # before any code generation so z_0 contains grounded knowledge
        # about package versions, API compatibility, and hardware.
        if not self.cognitive_state.environment_context:
            recon_summary = self.run_environment_recon(state)
            if recon_summary:
                self.cognitive_state.environment_context = recon_summary

        # Initialise CS tree root with a fresh z_0 (now includes recon)
        if self.cs_root is None:
            self.cs_root = CognitiveStateNode(
                cognitive_state=self.cognitive_state.clone(),
                depth=0,
            )

        while self.state.current_step < self.cfg.step_limit:
            start_time = time.monotonic()
            state = self._tree_step(task, state)
            self.state.running_time += time.monotonic() - start_time

            stats = tree_stats(self.cs_root)
            self.logger.info(
                f"MC-ESES step done: journal_step={self.state.current_step}, "
                f"elapsed={self.state.running_time:.1f}s, "
                f"cs_tree_nodes={stats['total_nodes']}, "
                f"cs_tree_depth={stats['max_depth']}, "
                f"cs_leaves={stats['leaf_count']}"
            )

            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs:
                self.logger.info("Maximum runtime reached, stopping MC-ESES search")
                break

        best_node = self.journal.get_best_node()

        try:
            export_search_results(self.cfg, self.journal, self.logger, "MC-ESES")
        except Exception as e:
            self.logger.error(f"Error exporting search results: {e}")

        if best_node:
            return state, best_node.code, best_node
        else:
            self.logger.info("No suitable code found after MC-ESES search.")
            return state, None, None

    def _tree_step(self, task, state):
        """One MC-ESES iteration: select → expand → backprop."""
        if not self.journal.nodes or self.data_preview_str is None:
            self.update_data_preview(state)

        # 1. Selection: UCT traversal from CS root to leaf
        path = self._cs_search_policy(self.cs_root)

        # 2. Expansion + backpropagation
        state = self._cs_expand_and_backprop(path, state, task)

        return state

    # ------------------------------------------------------------------
    # MC-ESES: selection
    # ------------------------------------------------------------------

    def _cs_search_policy(self, root: CognitiveStateNode) -> List[CognitiveStateNode]:
        """Traverse the cognitive state tree from root to leaf using UCT."""
        path: List[CognitiveStateNode] = []
        current = root
        intrinsic_w = self.cfg.intrinsic_quality_weight
        while True:
            path.append(current)
            if current.is_leaf:
                return path
            # Depth guard: treat max-depth nodes as leaves
            if current.depth >= self.cfg.max_cs_tree_depth:
                return path
            current = max(
                current.children,
                key=lambda c: uct_value(
                    q_value=c.q_value(self.lower_is_better),
                    explore_count=c.explore_count,
                    parent_explore_count=current.explore_count,
                    uct_c=self.cfg.uct_c,
                    global_max_q_val=self.state.global_max_q_val,
                    global_min_q_val=self.state.global_min_q_val,
                    intrinsic_quality=c.cognitive_state.intrinsic_quality(),
                    intrinsic_weight=intrinsic_w,
                ),
            )

    # ------------------------------------------------------------------
    # MC-ESES: expansion + backpropagation
    # ------------------------------------------------------------------

    def _cs_expand_and_backprop(
        self, path: List[CognitiveStateNode], state, task
    ):
        """Expand the selected CS leaf: generate code, execute, reflect, backprop.

        Integrates all Phase B algorithms:
        - Strategy-driven action selection (draft vs improve based on z_t.confidence)
        - Expansion diversity (assign different focus_directions to each child)
        - Cross-branch knowledge transfer (harvest insights from top-k leaves)
        - Multi-dimensional backpropagation (metric + validity + improvement)
        """
        leaf_cs = path[-1]
        target_children = (
            self._adaptive_num_children(leaf_cs.cognitive_state)
            if self.cfg.managed_evolution
            else self.cfg.num_children
        )
        num_children = min(
            target_children,
            self.cfg.step_limit - self.state.current_step,
        )

        # --- Cross-branch insights (P4) ---
        cross_branch_insights = None
        if self.cfg.crossover_enabled and self.cs_root is not None:
            cross_branch_insights = self._get_cross_branch_insights(leaf_cs)

        # --- Diverse expansion directions (P1) ---
        focus_directions = self._generate_diverse_directions(
            leaf_cs, num_children
        ) if self.cfg.diverse_expansion else [None] * num_children

        # Track expansion results for multi-dimensional backprop
        best_child_metric: Optional[float] = None
        prev_best_metric = self._get_current_best_metric()
        valid_count = 0
        total_count = 0

        for k in range(num_children):
            if self.state.current_step >= self.cfg.step_limit:
                break
            if self.state.running_time >= self.cfg.time_limit_secs:
                break

            total_count += 1

            # 1. Clone the leaf's cognitive state for this child
            z_t = leaf_cs.cognitive_state.clone()

            # 2. Strategy-driven action selection (P0): confidence → draft vs improve
            focus_dir = focus_directions[k] if k < len(focus_directions) else None
            code_node = self._select_and_execute_action(
                z_t,
                focus_dir,
                cross_branch_insights,
                branch_source_step=leaf_cs.source_node_step,
            )

            # 3. Execute
            state, eval_result = task.step_task(state, extract_code(code_node.code))
            self.parse_eval_result(node=code_node, eval_result=eval_result)
            self.journal.append(code_node)
            self.state.current_step += 1

            # 4. If buggy, try bounded debug cycle
            if code_node.is_buggy:
                state, code_node = self._cs_debug_cycle(state, task, code_node, z_t)

            # 5. Construct multi-dimensional feedback r_t
            feedback = build_feedback(code_node, self.journal, self.lower_is_better)
            if feedback is not None:
                code_node._term_out.append(f"\n\n{feedback.to_prompt_str()}")

            # 6. Managed state evolution: reflect only on informative events.
            z_before = z_t.clone()  # snapshot for trajectory
            should_reflect, reflect_trigger = self._should_reflect(z_t, feedback)
            if should_reflect:
                updated_state, reflect_metrics = self.reflect_fn(
                    self.task_desc,
                    z_t,
                    code_node,
                    self.journal,
                )
                metrics_to_log = dict(reflect_metrics or {})
                metrics_to_log["trigger"] = reflect_trigger
                code_node.operators_used.append("reflect")
                code_node.operators_metrics.append(metrics_to_log)
            else:
                updated_state = self._lightweight_state_update(
                    z_t,
                    code_node,
                    feedback,
                    reflect_trigger,
                )
                code_node.operators_used.append("reflect_skipped")
                code_node.operators_metrics.append(
                    {"trigger": reflect_trigger, "mode": "lightweight"}
                )

            updated_state = self._apply_managed_decay(
                z_before,
                updated_state,
                feedback,
                reflect_trigger,
            )

            # 6b. Causal intervention (Exp 3)
            updated_state = self._apply_intervention(
                updated_state, self.state.current_step
            )

            # 6c. Record trajectory
            if feedback is not None:
                self._record_step(
                    step=self.state.current_step - 1,
                    z_before=z_before,
                    z_after=updated_state,
                    feedback=feedback,
                    action=code_node.operators_used[0] if code_node.operators_used else "unknown",
                    code_node=code_node,
                )

            # 7. Create child CS node
            child_cs = CognitiveStateNode(
                cognitive_state=updated_state,
                parent=leaf_cs,
                source_node_step=code_node.step,
                depth=leaf_cs.depth + 1,
            )
            child_cs.explore_count = 1
            leaf_cs.children.append(child_cs)

            # 7. Track metrics for multi-dimensional backprop
            is_valid = not code_node.is_buggy and code_node.metric.value is not None
            if is_valid:
                valid_count += 1
                child_metric = code_node.metric.value
                child_cs.node_value = child_metric
                if best_child_metric is None or self._is_better(child_metric, best_child_metric):
                    best_child_metric = child_metric
            else:
                child_cs.node_value = self._buggy_child_value()

            # Log
            self._log_cs_expansion(k, leaf_cs, child_cs, code_node)

        # 8. Multi-dimensional backpropagation (P2)
        composite_value = self._compute_expansion_value(
            best_child_metric, prev_best_metric, valid_count, total_count
        )
        if composite_value is not None:
            self._cs_backprop(path, composite_value, valid_count, total_count)
            if best_child_metric is not None:
                self._set_global_q_values(best_child_metric)

        # Update the "current" cognitive state to the best leaf for prompt context
        self._update_current_cognitive_state()

        return state

    def _draft_from_state(
        self,
        z_t: CognitiveState,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
    ) -> Node:
        """Generate code from a specific cognitive state (temporarily swap context)."""
        saved = self.cognitive_state
        self.cognitive_state = z_t
        try:
            return self._draft(
                focus_direction=focus_direction,
                cross_branch_insights=cross_branch_insights,
            )
        finally:
            self.cognitive_state = saved

    def _improve_from_state(
        self,
        z_t: CognitiveState,
        parent_node: Node,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
    ) -> Node:
        """Improve code from a specific cognitive state (temporarily swap context)."""
        saved = self.cognitive_state
        self.cognitive_state = z_t
        try:
            return self._improve(
                parent_node,
                focus_direction=focus_direction,
                cross_branch_insights=cross_branch_insights,
            )
        finally:
            self.cognitive_state = saved

    def _debug_from_state(self, z_t: CognitiveState, parent_node: Node) -> Node:
        """Debug code using a specific branch-local cognitive state."""
        saved = self.cognitive_state
        self.cognitive_state = z_t
        try:
            return self._debug(parent_node)
        finally:
            self.cognitive_state = saved

    def _cs_debug_cycle(self, state, task, buggy_node: Node, z_t: CognitiveState) -> Tuple:
        """Bounded debug cycle for a buggy code node during CS expansion."""
        debug_start = time.monotonic()
        current_node = buggy_node

        for _ in range(self.cfg.max_debug_depth):
            if self.state.current_step >= self.cfg.step_limit:
                break
            if (time.monotonic() - debug_start) >= self.cfg.max_debug_time:
                break
            if current_node.debug_depth >= self.cfg.max_debug_depth:
                break

            debug_node = self._debug_from_state(z_t, current_node)
            state, eval_result = task.step_task(state, extract_code(debug_node.code))
            self.parse_eval_result(node=debug_node, eval_result=eval_result)
            self.journal.append(debug_node)
            self.state.current_step += 1

            if not debug_node.is_buggy:
                return state, debug_node

            current_node = debug_node

        return state, current_node

    # ------------------------------------------------------------------
    # MC-ESES: backpropagation
    # ------------------------------------------------------------------

    def _cs_backprop(
        self,
        path: List[CognitiveStateNode],
        value: float,
        valid_count: int = 0,
        total_count: int = 0,
    ) -> None:
        """Backpropagate composite value and validity stats up the CS tree path."""
        for cs_node in path:
            cs_node.increment_explore_count()
            cs_node.add_value(value)
            # Multi-dimensional tracking (P2)
            cs_node.validity_count += valid_count
            cs_node.total_expansions += total_count

    def _set_global_q_values(self, metric_value: float) -> None:
        """Update global Q-value bounds for UCT normalisation."""
        self.state.global_max_q_val = max(self.state.global_max_q_val, metric_value)
        self.state.global_min_q_val = min(self.state.global_min_q_val, metric_value)

    def _is_better(self, a: float, b: float) -> bool:
        """Compare two metric values respecting optimisation direction."""
        return a < b if self.lower_is_better else a > b

    def _update_current_cognitive_state(self) -> None:
        """Set self.cognitive_state to the best CS leaf (highest Q-value).

        This ensures that if Phase A methods are called (e.g. during logging),
        the current cognitive state reflects the best trajectory found so far.
        """
        if self.cs_root is None:
            return
        best_node = self.cs_root
        queue = [self.cs_root]
        best_q = -1e8
        while queue:
            node = queue.pop()
            q = node.q_value(self.lower_is_better)
            if q > best_q and node.explore_count > 0:
                best_q = q
                best_node = node
            queue.extend(node.children)
        self.cognitive_state = best_node.cognitive_state

    def _get_branch_anchor_node(self, source_step: Optional[int]) -> Optional[Node]:
        """Resolve the Journal node that anchors the current branch."""
        if source_step is None:
            return None
        if source_step < 0 or source_step >= len(self.journal.nodes):
            return None
        anchor = self.journal.nodes[source_step]
        if anchor.is_buggy:
            return None
        return anchor

    def _buggy_child_value(self) -> float:
        """Return a pessimistic value for a visited-but-buggy child."""
        hi = self.state.global_max_q_val
        lo = self.state.global_min_q_val
        if hi <= lo:
            return 1e8 if self.lower_is_better else -1e8
        margin = 0.05 * abs(hi - lo) + 1.0
        return (hi + margin) if self.lower_is_better else (lo - margin)

    def _adaptive_num_children(self, z_t: CognitiveState) -> int:
        """Adapt branching factor to branch confidence."""
        if not (self.cfg.managed_evolution and self.cfg.adaptive_children):
            return self.cfg.num_children

        if z_t.confidence < self.cfg.low_conf_threshold:
            target = self.cfg.low_conf_children
        elif z_t.confidence >= self.cfg.high_conf_threshold:
            target = self.cfg.high_conf_children
        else:
            target = self.cfg.mid_conf_children

        return max(1, min(self.cfg.num_children, target))

    def _recent_bug_streak(self, z_t: CognitiveState, current_feedback: Optional[Feedback] = None) -> int:
        """Count consecutive buggy attempts for the current branch."""
        streak = 0

        if current_feedback is not None:
            if not current_feedback.is_buggy:
                return 0
            streak += 1

        for attempt in reversed(z_t.attempt_summaries):
            if attempt.is_buggy:
                streak += 1
            else:
                break

        return streak

    def _recent_valid_metrics(self, z_t: CognitiveState) -> List[float]:
        """Return valid historical metrics for the current branch."""
        return [
            attempt.metric
            for attempt in z_t.attempt_summaries
            if not attempt.is_buggy and attempt.metric is not None
        ]

    def _is_plateau(self, z_t: CognitiveState, current_metric: Optional[float]) -> bool:
        """Detect whether the branch has plateaued over the recent valid attempts."""
        if current_metric is None:
            return False

        metrics = self._recent_valid_metrics(z_t)
        metrics.append(current_metric)
        window = metrics[-self.cfg.plateau_window :]
        if len(window) < self.cfg.plateau_window:
            return False

        best_so_far = window[0]
        for metric in window[1:]:
            if self._is_better(metric, best_so_far):
                return False
        return True

    def _get_reflection_trigger(self, z_t: CognitiveState, feedback: Feedback) -> str:
        """Classify whether the current step contains enough new information to reflect."""
        if z_t.evolution_step == 0 or not z_t.attempt_summaries:
            return "bootstrap"

        previous_metrics = self._recent_valid_metrics(z_t)
        if not feedback.is_buggy and not previous_metrics:
            return "first_success"

        if feedback.is_buggy and feedback.error_category and feedback.error_category != "none":
            if feedback.error_category not in self._prev_error_categories:
                return "new_error"

        if feedback.metric is not None and previous_metrics:
            prev_metric = previous_metrics[-1]
            denom = max(abs(prev_metric), 1e-8)
            rel_change = abs(feedback.metric - prev_metric) / denom
            if rel_change > self.cfg.score_jump_threshold:
                return "score_jump"

        if self._recent_bug_streak(z_t, feedback) >= self.cfg.bug_streak_window:
            return "bug_streak"

        if self._is_plateau(z_t, feedback.metric):
            return "plateau"

        return "routine"

    def _should_reflect(self, z_t: CognitiveState, feedback: Feedback) -> Tuple[bool, str]:
        """Decide whether to spend an LLM call on reflect_op for this child."""
        if not self.cfg.managed_evolution:
            return self.cfg.reflect_after_every_step, "always"

        if not self.cfg.triggered_reflection:
            return self.cfg.reflect_after_every_step, "always"

        trigger = self._get_reflection_trigger(z_t, feedback)
        if trigger != "routine":
            return True, trigger
        return self.cfg.reflect_on_routine, trigger

    def _clamp_confidence(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _lightweight_state_update(
        self,
        z_t: CognitiveState,
        code_node: Node,
        feedback: Feedback,
        trigger: str,
    ) -> CognitiveState:
        """Advance state without an LLM call when the step contains little new information."""
        updated = z_t.clone()
        updated.evolution_step += 1

        if feedback.is_buggy:
            updated.confidence = self._clamp_confidence(updated.confidence - 0.05)
            if feedback.error_pattern and feedback.error_pattern not in updated.learned_patterns:
                updated.learned_patterns.append(feedback.error_pattern)
            insight = feedback.error_pattern or f"buggy step ({feedback.error_category or 'unknown error'})"
        else:
            if feedback.trend == "improving":
                updated.confidence = self._clamp_confidence(updated.confidence + 0.05)
            elif feedback.trend == "degrading":
                updated.confidence = self._clamp_confidence(updated.confidence - 0.05)
            else:
                updated.confidence = self._clamp_confidence(updated.confidence + 0.01)
            insight = f"{feedback.trend} metric ({trigger})"

        updated.add_attempt(
            step=z_t.evolution_step,
            approach=code_node.plan or "(unknown)",
            metric=feedback.metric,
            is_buggy=feedback.is_buggy,
            key_insight=insight,
            max_history=self.cfg.max_state_history,
        )

        return updated

    def _apply_managed_decay(
        self,
        z_before: CognitiveState,
        z_after: CognitiveState,
        feedback: Feedback,
        trigger: str,
    ) -> CognitiveState:
        """Apply conservative decay so unhealthy branches let go earlier."""
        if not self.cfg.managed_evolution:
            return z_after

        updated = z_after.clone()
        bug_streak = self._recent_bug_streak(z_before, feedback)

        if self.cfg.hard_reset_on_bug_streak and bug_streak >= self.cfg.bug_streak_window:
            updated.confidence = 0.1
            updated.preferred_directions = []
            if len(updated.hypotheses) > 2:
                updated.hypotheses = updated.hypotheses[:2]
            return updated

        if self.cfg.state_decay_on_plateau and trigger == "plateau":
            updated.confidence = self._clamp_confidence(
                updated.confidence * (1.0 - self.cfg.plateau_decay_rate)
            )
            if updated.preferred_directions:
                updated.preferred_directions = updated.preferred_directions[:-1]

        return updated

    def _log_cs_expansion(
        self,
        child_idx: int,
        parent_cs: CognitiveStateNode,
        child_cs: CognitiveStateNode,
        code_node: Node,
    ) -> None:
        """Log a single CS tree expansion."""
        metric_str = "buggy" if code_node.is_buggy else f"{code_node.metric.value}"
        best_node = self.journal.get_best_node()
        best_step = best_node.step if best_node else 0

        self.logger.info(
            f"MC-ESES expand[{child_idx}]: "
            f"parent_depth={parent_cs.depth}, child_depth={child_cs.depth}, "
            f"metric={metric_str}, "
            f"z_t.confidence={child_cs.cognitive_state.confidence:.2f}, "
            f"z_t.evolution_step={child_cs.cognitive_state.evolution_step}, "
            f"journal_best_step={best_step}"
        )
        self.logger.log(
            self.journal.get_node_data(code_node.step)
            | {"current_best_node": best_step},
            "JOURNAL",
            step=code_node.step,
        )
        self.logger.log(self.state.state_dict(), "STATE", step=self.state.current_step)

    # ------------------------------------------------------------------
    # MC-ESES: strategy-driven action selection (P0)
    # ------------------------------------------------------------------

    def _select_and_execute_action(
        self,
        z_t: CognitiveState,
        focus_direction: Optional[str] = None,
        cross_branch_insights: Optional[Dict] = None,
        branch_source_step: Optional[int] = None,
    ) -> Node:
        """Select draft vs improve based on z_t.confidence and journal state.

        Strategy-driven action selection (P0):
        - Low confidence (< 0.4) or few drafts → draft (explore new approaches)
        - High confidence (>= 0.4) with good nodes → improve (exploit best solution)
        """
        if not self.cfg.strategy_driven:
            return self._draft_from_state(z_t, focus_direction, cross_branch_insights)

        # Always draft if we don't have enough initial drafts
        if len(self.journal.draft_nodes) < self.cfg.num_drafts:
            return self._draft_from_state(z_t, focus_direction, cross_branch_insights)

        # Use branch-local anchor for improve. If this branch has not yet produced
        # a valid artifact to refine, fall back to draft instead of improving a
        # global best node from another branch.
        anchor_node = self._get_branch_anchor_node(branch_source_step)
        candidate_direction = focus_direction
        if (
            self.cfg.state_guided_policy
            and candidate_direction is not None
            and candidate_direction in z_t.avoided_directions
        ):
            candidate_direction = None

        if self.cfg.state_guided_policy:
            if anchor_node is None:
                self.logger.info(
                    f"Strategy: DRAFT (no branch-local anchor, confidence={z_t.confidence:.2f})"
                )
                return self._draft_from_state(z_t, candidate_direction, cross_branch_insights)

            if z_t.confidence < self.cfg.low_conf_threshold:
                self.logger.info(
                    f"Strategy: DRAFT (low confidence={z_t.confidence:.2f} < {self.cfg.low_conf_threshold:.2f})"
                )
                return self._draft_from_state(z_t, candidate_direction, cross_branch_insights)

            if candidate_direction is not None and z_t.confidence < self.cfg.high_conf_threshold:
                self.logger.info(
                    f"Strategy: DRAFT (pursuing focus_direction with medium confidence={z_t.confidence:.2f})"
                )
                return self._draft_from_state(z_t, candidate_direction, cross_branch_insights)

            if self.cfg.avoid_dead_end_improve and anchor_node.plan:
                anchor_plan = anchor_node.plan.lower()
                if any(
                    dead_end.lower() in anchor_plan
                    for dead_end in z_t.avoided_directions
                    if dead_end
                ):
                    self.logger.info(
                        "Strategy: DRAFT (branch anchor overlaps an avoided direction)"
                    )
                    return self._draft_from_state(z_t, candidate_direction, cross_branch_insights)

        if z_t.confidence >= 0.4 and anchor_node is not None:
            self.logger.info(
                f"Strategy: IMPROVE (confidence={z_t.confidence:.2f} >= 0.4, "
                f"branch_source_step={branch_source_step})"
            )
            return self._improve_from_state(
                z_t, anchor_node, candidate_direction, cross_branch_insights
            )
        else:
            self.logger.info(
                f"Strategy: DRAFT (confidence={z_t.confidence:.2f}, "
                f"branch_source_step={branch_source_step}, "
                f"anchor_valid={anchor_node is not None})"
            )
            return self._draft_from_state(z_t, candidate_direction, cross_branch_insights)

    # ------------------------------------------------------------------
    # MC-ESES: expansion diversity (P1)
    # ------------------------------------------------------------------

    def _generate_diverse_directions(
        self, leaf_cs: CognitiveStateNode, num_children: int
    ) -> List[Optional[str]]:
        """Generate diverse focus directions for each expansion child.

        Uses the cognitive state's hypotheses and preferred_directions
        as a pool, distributing one per child. If the pool is smaller
        than num_children, remaining children get None (free exploration).
        """
        z_t = leaf_cs.cognitive_state
        direction_pool: List[str] = []

        # Gather from hypotheses and preferred_directions
        direction_pool.extend(z_t.hypotheses)
        direction_pool.extend(z_t.preferred_directions)

        # Deduplicate while preserving order
        seen = set()
        unique_pool: List[str] = []
        for d in direction_pool:
            if d not in seen:
                seen.add(d)
                unique_pool.append(d)

        # Remove any directions that are in the avoided list
        avoided_set = set(z_t.avoided_directions)
        unique_pool = [d for d in unique_pool if d not in avoided_set]

        # Assign one direction per child, cycling if pool is large enough
        result: List[Optional[str]] = []
        for k in range(num_children):
            if unique_pool:
                result.append(unique_pool[k % len(unique_pool)])
            else:
                result.append(None)

        return result

    # ------------------------------------------------------------------
    # MC-ESES: multi-dimensional backprop value (P2)
    # ------------------------------------------------------------------

    def _compute_expansion_value(
        self,
        best_metric: Optional[float],
        prev_best_metric: Optional[float],
        valid_count: int,
        total_count: int,
    ) -> Optional[float]:
        """Compute composite backprop value from expansion results.

        Combines three signals:
        - α · metric:      raw best metric from this expansion
        - β · validity:    fraction of non-buggy children
        - γ · improvement: whether this expansion improved on global best

        Returns None if no valid results at all.
        """
        if total_count == 0:
            return None

        alpha = self.cfg.backprop_metric_weight
        beta = self.cfg.backprop_validity_weight
        gamma = self.cfg.backprop_improvement_weight

        # Metric component: use best_metric if available, else 0
        metric_component = best_metric if best_metric is not None else 0.0

        # Validity component: fraction of non-buggy expansions, scaled to [0, 1]
        validity_rate = valid_count / total_count

        # Improvement component: did this expansion improve on previous best?
        if best_metric is not None and prev_best_metric is not None:
            improved = self._is_better(best_metric, prev_best_metric)
            improvement_signal = 1.0 if improved else 0.0
        elif best_metric is not None:
            # First valid result is always an improvement
            improvement_signal = 1.0
        else:
            improvement_signal = 0.0

        # Composite: α·metric + β·validity + γ·improvement
        # Note: metric_component is in raw metric scale; validity and improvement
        # are in [0,1]. We scale validity and improvement by the metric magnitude
        # to keep units consistent.
        if best_metric is not None:
            composite = (
                alpha * metric_component
                + beta * validity_rate * abs(metric_component)
                + gamma * improvement_signal * abs(metric_component)
            )
        else:
            # No valid metric at all — return a small penalty
            composite = 0.0

        return composite

    def _get_current_best_metric(self) -> Optional[float]:
        """Get the current best metric from the journal."""
        best_node = self.journal.get_best_node()
        if best_node is not None and not best_node.is_buggy and best_node.metric.value is not None:
            return best_node.metric.value
        return None

    # ------------------------------------------------------------------
    # MC-ESES: cross-branch cognitive crossover (P4)
    # ------------------------------------------------------------------

    def _get_cross_branch_insights(
        self, current_leaf: CognitiveStateNode
    ) -> Optional[Dict[str, Any]]:
        """Harvest insights from top-k leaves across the CS tree.

        Collects learned_patterns and avoided_directions from the best
        leaves (by Q-value) that are NOT ancestors/descendants of current_leaf.
        Returns a dict suitable for injection into cognitive state prompts.
        """
        if self.cs_root is None:
            return None

        # Collect all leaves with their Q-values
        leaves: List[Tuple[float, CognitiveStateNode]] = []
        queue: List[CognitiveStateNode] = [self.cs_root]
        while queue:
            node = queue.pop()
            if node.is_leaf and node.explore_count > 0:
                leaves.append((node.q_value(self.lower_is_better), node))
            queue.extend(node.children)

        if not leaves:
            return None

        # Sort by Q-value descending (best first)
        leaves.sort(key=lambda x: x[0], reverse=True)

        # Get ancestors of current_leaf to exclude same-branch nodes
        ancestors = set()
        p = current_leaf
        while p is not None:
            ancestors.add(id(p))
            p = p.parent

        # Harvest from top-k leaves that are on different branches
        cross_patterns: List[str] = []
        dead_ends: List[str] = []
        harvested = 0

        for _, leaf_node in leaves:
            if harvested >= self.cfg.crossover_top_k:
                break
            if id(leaf_node) in ancestors:
                continue

            z = leaf_node.cognitive_state
            for pattern in z.learned_patterns:
                if pattern not in cross_patterns:
                    cross_patterns.append(pattern)
            for avoid in z.avoided_directions:
                if avoid not in dead_ends:
                    dead_ends.append(avoid)
            harvested += 1

        if not cross_patterns and not dead_ends:
            return None

        return {
            "cross_branch_patterns": cross_patterns[:10],
            "confirmed_dead_ends": dead_ends[:5],
        }

    # ------------------------------------------------------------------
    # Utilities (inherited from Greedy pattern)
    # ------------------------------------------------------------------

    def discover_packages(self, state):
        if "solver_interpreter" not in state:
            self.logger.warning("No interpreter — using static package list")
            return

        code = (
            "import subprocess, sys\n"
            "r = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],\n"
            "                   capture_output=True, text=True)\n"
            "print(r.stdout)\n"
        )

        try:
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            output = "\n".join(exec_result.term_out)
            packages = parse_pip_list_output(output)
        except Exception as e:
            self.logger.warning(f"pip list failed ({e}) — using static package list")
            return

        if not packages:
            return

        self.logger.info(f"Discovered {len(packages)} packages")
        try:
            from omegaconf import OmegaConf
            OmegaConf.update(self.cfg, "available_packages", packages, force_add=True)
        except Exception:
            try:
                self.cfg.available_packages = packages
            except Exception as e2:
                self.logger.warning(f"Could not update packages ({e2})")

    # ------------------------------------------------------------------
    # Environment reconnaissance
    # ------------------------------------------------------------------

    _RECON_SCRIPT = '''\
import sys, os, json

report = {}

# 1. Python version
report["python"] = sys.version.split()[0]

# 2. Key package versions & API compatibility
compatibility = []
warnings = []

try:
    import torch
    tv = torch.__version__
    report["torch"] = tv
    # Check ReduceLROnPlateau(verbose) removal (PyTorch >= 2.4)
    major, minor = int(tv.split(".")[0]), int(tv.split(".")[1])
    if major >= 2 and minor >= 4:
        warnings.append("ReduceLROnPlateau: 'verbose' argument REMOVED in PyTorch>=2.4. Do NOT pass verbose=True.")
    # GPU info
    if torch.cuda.is_available():
        report["gpu_count"] = torch.cuda.device_count()
        report["gpu_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        report["gpu_memory_gb"] = round(mem / 1e9, 1)
        # bf16 support
        if torch.cuda.get_device_capability(0)[0] >= 8:
            compatibility.append("bf16 AMP: SUPPORTED (compute capability >= 8.0)")
        else:
            warnings.append("bf16 AMP: NOT supported on this GPU. Use fp16=True instead.")
    else:
        report["gpu_count"] = 0
        warnings.append("No GPU available. Use CPU-only methods.")
except ImportError:
    report["torch"] = "NOT INSTALLED"

try:
    import transformers
    tv = transformers.__version__
    report["transformers"] = tv
    major, minor = int(tv.split(".")[0]), int(tv.split(".")[1])
    if major >= 4 and minor >= 46:
        warnings.append("transformers>=4.46: 'predict_with_generate' REMOVED from TrainingArguments. Use GenerationConfig instead.")
except ImportError:
    pass

try:
    import evaluate
    report["evaluate"] = evaluate.__version__
except ImportError:
    warnings.append("'evaluate' library is NOT installed. Compute metrics manually (e.g. sklearn.metrics).")

try:
    import datasets
    report["datasets"] = datasets.__version__
    warnings.append("datasets.map(): Always use num_proc=1 to avoid subprocess crashes in sandboxed environments.")
except ImportError:
    pass

try:
    import sklearn
    report["scikit-learn"] = sklearn.__version__
except ImportError:
    pass

try:
    import lightgbm
    report["lightgbm"] = lightgbm.__version__
    warnings.append("lightgbm: Use lightgbm.early_stopping(stopping_rounds=N) callback instead of deprecated early_stopping_rounds argument.")
except ImportError:
    pass

try:
    import xgboost
    report["xgboost"] = xgboost.__version__
except ImportError:
    pass

# 3. Hardware
report["cpu_count"] = os.cpu_count()

# 4. Data directory structure
data_dir = "./data"
if os.path.isdir(data_dir):
    data_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files[:50]:
            rel = os.path.relpath(os.path.join(root, f), data_dir)
            size = os.path.getsize(os.path.join(root, f))
            data_files.append(f"{rel} ({size} bytes)")
    report["data_files"] = data_files[:30]

report["compatibility"] = compatibility
report["warnings"] = warnings

print("===RECON_JSON_START===")
print(json.dumps(report, indent=2))
print("===RECON_JSON_END===")
'''

    def run_environment_recon(self, state) -> str:
        """Execute a lightweight reconnaissance script in the execution
        environment and return a structured summary for injection into z_0.

        The script probes:
        - Key package versions (torch, transformers, sklearn, etc.)
        - API compatibility warnings (deprecated/removed arguments)
        - GPU hardware (count, name, memory, bf16 support)
        - Data directory structure
        """
        if "solver_interpreter" not in state:
            self.logger.warning("No interpreter — skipping environment recon")
            return ""

        try:
            exec_result = state["solver_interpreter"].run(
                self._RECON_SCRIPT, include_exec_time=False,
            )
            raw = "\n".join(exec_result.term_out)
        except Exception as e:
            self.logger.warning(f"Environment recon failed ({e})")
            return ""

        # Extract JSON from markers
        import json as _json
        start = raw.find("===RECON_JSON_START===")
        end = raw.find("===RECON_JSON_END===")
        if start < 0 or end < 0:
            self.logger.warning("Recon script produced no parseable output")
            return raw[:2000] if raw else ""

        json_str = raw[start + len("===RECON_JSON_START==="):end].strip()
        try:
            report = _json.loads(json_str)
        except _json.JSONDecodeError:
            self.logger.warning("Recon JSON parse failed")
            return json_str[:2000]

        # Format into a concise human-readable summary
        lines = []
        lines.append(f"Python {report.get('python', '?')}")

        # Package versions
        pkg_keys = ["torch", "transformers", "scikit-learn", "lightgbm",
                     "xgboost", "datasets", "evaluate"]
        pkg_parts = []
        for k in pkg_keys:
            if k in report:
                pkg_parts.append(f"{k}=={report[k]}")
        if pkg_parts:
            lines.append(f"Key packages: {', '.join(pkg_parts)}")

        # Hardware
        gpu_count = report.get("gpu_count", 0)
        if gpu_count:
            lines.append(
                f"GPU: {gpu_count}x {report.get('gpu_name', '?')} "
                f"({report.get('gpu_memory_gb', '?')} GB)"
            )
        else:
            lines.append("GPU: none (CPU only)")
        lines.append(f"CPUs: {report.get('cpu_count', '?')}")

        # Compatibility notes
        for c in report.get("compatibility", []):
            lines.append(f"[OK] {c}")

        # Warnings — these are the most valuable part
        for w in report.get("warnings", []):
            lines.append(f"[WARNING] {w}")

        # Data files (brief)
        data_files = report.get("data_files", [])
        if data_files:
            lines.append(f"Data directory: {len(data_files)} files")
            for f in data_files[:10]:
                lines.append(f"  - {f}")
            if len(data_files) > 10:
                lines.append(f"  ... and {len(data_files) - 10} more")

        summary = "\n".join(lines)
        self.logger.info(f"Environment recon complete:\n{summary}")
        return summary

    def update_data_preview(self, state):
        assert "solver_interpreter" in state
        if state["solver_interpreter"].local:
            self.data_preview_str = data_preview.generate(
                state["solver_interpreter"].working_dir
            )
        else:
            import inspect
            path = Path(inspect.getsourcefile(data_preview))
            script = path.read_text()
            code = f"{script}\nprint(generate(Path('.').resolve()))"
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            self.data_preview_str = "\n".join(exec_result.term_out)
