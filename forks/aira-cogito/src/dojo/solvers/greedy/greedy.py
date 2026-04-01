# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import hydra
from dojo.core.solvers.base import Solver
from dojo.core.solvers.operators.analyze import analyze_op
from dojo.core.solvers.operators.core import execute_op_plan_code
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.utils import data_preview
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.solvers.utils import get_complextiy_level
from dojo.utils.code_parsing import parse_json_output
from dojo.core.solvers.utils.search_exporter import (
    export_search_results,
)
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
import time

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM

from dojo.config_dataclasses.solver.greedy import GreedySolverConfig
from dojo.utils.environment import parse_pip_list_output
from dojo.utils.state import GreedyState

# Cognitive state evolution imports (used when use_cognitive_state=True)
from dojo.core.solvers.utils.cognitive_state import (
    CognitiveState,
    Feedback,
    build_feedback,
)
from dojo.core.solvers.operators.reflect import reflect_op
from dojo.solvers.mceses.instrumentation import (
    classify_trigger,
    compute_state_delta,
    save_trajectory_snapshot,
)


class Greedy(Solver):
    """Greedy solver."""

    def __init__(self, cfg: GreedySolverConfig, task_info):
        """
        Initialize the Greedy solver.

        Args:
            task_info: Dictionary containing task information including description and optimization direction
            **cfg: Configuration dictionary with solver parameters
        """
        super().__init__(cfg, task_info=task_info)
        self.journal = Journal()
        self.data_preview: str | None = None

        self.task_desc = task_info[TASK_DESCRIPTION]
        self.lower_is_better = task_info.get("lower_is_better", None)

        assert self.lower_is_better is not None

        self.setup_operators()

        self.state = GreedyState()

        # --- Cognitive state evolution (optional) ---
        if self.cfg.use_cognitive_state:
            self.cognitive_state = CognitiveState()
            self._trajectory_path: Optional[Path] = None
            self._frozen_state: Optional[CognitiveState] = None
            self._scrambled_state: Optional[CognitiveState] = None
            # Tracking for trigger classification
            self._prev_error_categories: set = set()
            self._recent_metrics: List[float] = []
            self._has_any_valid: bool = False
            # Load scrambled state if needed
            if self.cfg.intervention_mode == "scrambled" and self.cfg.scramble_source_path:
                try:
                    with open(self.cfg.scramble_source_path) as f:
                        self._scrambled_state = CognitiveState.from_dict(json.load(f))
                    self.logger.info(f"Loaded scrambled state from {self.cfg.scramble_source_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load scrambled state: {e}")

    def save_checkpoint(self):
        super().save_checkpoint()

        # Write the journal to a jsonl file
        journal_sd = self.journal.node_list()
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        with open(journal_path, "w") as f:
            for node in journal_sd:
                f.write(json.dumps(node) + "\n")

        # Save cognitive state if enabled
        if self.cfg.use_cognitive_state:
            cs_path = Path(self.cfg.checkpoint_path) / "cognitive_state.json"
            with open(cs_path, "w") as f:
                json.dump(self.cognitive_state.to_dict(), f, indent=2)

        self.logger.info(f"Checkpoint saved to {self.cfg.checkpoint_path}")

    def load_checkpoint(self):
        super().load_checkpoint()

        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        if not journal_path.exists():
            assert self.state.current_step == 0, (
                f"No journal found at {journal_path}, but the state was found. This is unexpected."
            )
            return

        self.logger.info(f"Found journal at {journal_path}. Loading...")
        with open(journal_path, "r") as f:
            journal_export = [json.loads(line) for line in f]
        self.journal = Journal.from_export_data({"nodes": journal_export})

        # Load cognitive state if enabled
        if self.cfg.use_cognitive_state:
            cs_path = Path(self.cfg.checkpoint_path) / "cognitive_state.json"
            if cs_path.exists():
                with open(cs_path, "r") as f:
                    self.cognitive_state = CognitiveState.from_dict(json.load(f))
                self.logger.info(f"Loaded cognitive state from {cs_path}")

    def setup_operators(self):
        """
        Initialize and configure the LLM operators used in the Greedy solver.

        This method instantiates the draft, improve, debug, and analyze LLMs
        and creates partial functions for each operator with the appropriate parameters.
        """

        # First we set up the LLMs
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        self.draft_fn = partial(draft_op, draft_llm, self.cfg, self.memory_op)
        self.improve_fn = partial(improve_op, improve_llm, self.cfg, self.memory_op)
        self.debug_fn = partial(debug_op, debug_llm, self.cfg, self.debug_memory_op)
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)

        # Reflect operator (only when cognitive state is enabled)
        if self.cfg.use_cognitive_state and "reflect" in self.cfg.operators:
            reflect_llm = GenericLLM(self.cfg.operators["reflect"])
            self.reflect_fn = partial(reflect_op, reflect_llm, self.cfg)

    def create_root_node(self):
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

    def __call__(self, task, state):
        """
        Run the Greedy solver for a specified number of iterations.

        Executes the search process for the configured number of steps, tracking the best
        solution found. After all iterations, exports search results and returns the best code.

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and the best code solution (or None if no valid solution found)
        """
        self.logger.info("Starting Greedy search")

        # Create a blank root node to start.
        self.create_root_node()

        # Step 0: discover actually installed packages inside the container
        self.discover_packages(state)

        # Cognitive state: environment recon + trajectory init
        if self.cfg.use_cognitive_state:
            if not self.cognitive_state.environment_context:
                recon_summary = self._run_environment_recon(state)
                if recon_summary:
                    self.cognitive_state.environment_context = recon_summary
            if self.cfg.save_trajectory and self._trajectory_path is None:
                self._trajectory_path = Path(self.cfg.checkpoint_path) / "trajectory.jsonl"

        # Run the search
        for _ in range(self.state.current_step, self.cfg.step_limit):
            start_time = time.monotonic()
            state, _ = self.step(task, state)
            self.state.running_time += time.monotonic() - start_time
            self.logger.info(
                f"Step {self.state.current_step}: Time taken for step: {self.state.running_time:.3f} seconds"
            )

            self.state.current_step += 1

            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        # Get the best node
        best_node = self.journal.get_best_node()

        # Export the search results
        try:
            export_search_results(self.cfg, self.journal, self.logger, "Greedy")
        except Exception as e:
            self.logger.error(f"Error exporting search results: {e}")

        # Return the best node
        if best_node:
            return state, best_node.code, best_node
        else:
            self.logger.info("No suitable code found after all iterations.")
            return state, None, None

    def search_policy(self) -> Node | None:
        """
        Determine the next node to work on based on the current state of the journal.

        This node selection policy determines whether to:
        - Draft a new solution (returns None)
        - Debug a buggy node (returns a buggy node)
        - Improve the best node (returns the best performing node)

        The decision is based on:
        1. Number of existing drafts compared to required drafts
        2. Random probability for debugging
        3. Existence of good (non-buggy) nodes to improve

        Returns:
            Node | None: Selected node to work on, or None to indicate a new draft should be created
        """
        # If not enough drafts exist, return None -> draft a new solution.
        if len(self.journal.draft_nodes) < self.cfg.num_drafts:
            self.logger.info(
                f"Search Policy: Drafting a new node (not enough drafts - {len(self.journal.draft_nodes)}/{self.cfg.num_drafts})"
            )
            return None

        # With probability debug_prob, try to debug a buggy node.
        if random.random() < self.cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes if (n.is_leaf and n.debug_depth <= self.cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                self.logger.info("Search Policy: Debugging a buggy node")
                return random.choice(debuggable_nodes)
            self.logger.debug("Search Policy: Not debugging (by random chance)")

        # If no good nodes exist, return None -> draft a new solution.
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            self.logger.info("Search Policy: Drafting a new node (no good nodes exist)")
            return None

        # Otherwise, pick the best node for improvement
        self.logger.info("Search Policy: Selecting best node for improvement")
        return self.journal.get_best_node()

    def _get_cognitive_context(self) -> Optional[str]:
        """Render cognitive state for prompt injection (None if disabled)."""
        if not self.cfg.use_cognitive_state:
            return None
        s = self.cognitive_state.to_prompt_str()
        return s if s else None

    def _draft(self) -> Node:
        """Generate a new solution from scratch using the draft LLM operator."""
        self.logger.info(f"Step {self.state.current_step}: Starting to drafting new solution")
        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            get_complextiy_level(num=len(self.journal.draft_nodes)) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(),
        )
        node = Node(
            plan=plan, code=code, operators_used=["draft"], operators_metrics=[metrics], parents=[self.root_node]
        )
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        return node

    def _improve(self, parent_node: Node) -> Node:
        """Improve an existing solution using the improve LLM operator."""
        self.logger.info(f"Step {self.state.current_step}: Starting to improve existing solution")
        plan, code, metrics = execute_op_plan_code(
            self.improve_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            get_complextiy_level(parent_node) if self.cfg.use_complexity else None,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(),
        )
        node = Node(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        return node

    def _debug(self, parent_node: Node) -> Node:
        """Debug a buggy solution using the debug LLM operator."""
        self.logger.info(f"Step {self.state.current_step}: Starting to debug buggy solution")
        plan, code, metrics = execute_op_plan_code(
            self.debug_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
            cognitive_state_str=self._get_cognitive_context(),
        )
        node = Node(plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics])
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        return node

    def _analyze(self, node: Node) -> Union[str, dict]:
        """
        Analyze a node's execution results using the analyze LLM operator.

        Processes the task description, code, and execution output to determine
        if the solution is buggy and to extract metrics when available.

        Args:
            node: The node to analyze

        Returns:
            Union[str, dict]: Analysis results, either as a string or dictionary
        """
        analysis, metrics = self.analyze_fn(self.task_desc, node)
        node.operators_used.append("analysis")
        node.operators_metrics.append(metrics)
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        self.logger.info(f"Step {self.state.current_step}: End of analyzing solution")
        return analysis

    # ------------------------------------------------------------------
    # Cognitive state evolution helpers
    # ------------------------------------------------------------------

    def _reflect_and_record(self, code_node: Node) -> None:
        """Reflect on execution results to update z_t, apply intervention, record trajectory."""
        # 1. Build multi-dimensional feedback r_t
        feedback = build_feedback(code_node, self.journal, self.lower_is_better)

        # 2. Snapshot z_before
        z_before = self.cognitive_state.clone()

        # 3. Determine whether to do full LLM reflect or lightweight update
        if self.cfg.intervention_mode == "ablated":
            # Ablated: skip reflect entirely, return blank state
            blank = CognitiveState()
            blank.environment_context = self.cognitive_state.environment_context
            updated_state = blank
            code_node.operators_used.append("reflect_skipped")
            code_node.operators_metrics.append({})
            trigger = "ablated"
        elif self.cfg.managed_evolution:
            # NAT+: triggered reflection
            if feedback is None:
                # Defensive: if feedback construction failed, fall back to full reflect
                updated_state, reflect_metrics = self.reflect_fn(
                    self.task_desc, self.cognitive_state, code_node, self.journal,
                )
                code_node.operators_used.append("reflect")
                code_node.operators_metrics.append(reflect_metrics)
                trigger = "fallback"
                self.cognitive_state = self._apply_intervention(updated_state, self.state.current_step)
                self.logger.warning("NAT+ fallback: feedback is None, did full reflect")
                return

            should_reflect, trigger = self._should_reflect(self.cognitive_state, feedback)
            if should_reflect:
                updated_state, reflect_metrics = self.reflect_fn(
                    self.task_desc, self.cognitive_state, code_node, self.journal,
                )
                code_node.operators_used.append("reflect")
                code_node.operators_metrics.append(reflect_metrics)
            else:
                updated_state = self._lightweight_state_update(
                    self.cognitive_state, code_node, feedback, trigger,
                )
                code_node.operators_used.append("reflect_lightweight")
                code_node.operators_metrics.append({})

            # Apply managed decay
            updated_state = self._apply_managed_decay(
                z_before, updated_state, feedback, trigger,
            )

            # Apply causal intervention
            updated_state = self._apply_intervention(
                updated_state, self.state.current_step,
            )
        else:
            # NAT (original): always reflect
            updated_state, reflect_metrics = self.reflect_fn(
                self.task_desc, self.cognitive_state, code_node, self.journal,
            )
            code_node.operators_used.append("reflect")
            code_node.operators_metrics.append(reflect_metrics)
            trigger = "always"

            # Apply causal intervention
            updated_state = self._apply_intervention(
                updated_state, self.state.current_step,
            )

        # 4. Update cognitive state
        self.cognitive_state = updated_state

        # 5. Record trajectory snapshot
        if feedback is not None:
            self._record_step(
                step=self.state.current_step,
                z_before=z_before,
                z_after=updated_state,
                feedback=feedback,
                action=code_node.operators_used[0] if code_node.operators_used else "unknown",
                code_node=code_node,
            )

        self.logger.info(
            f"Reflect[{trigger}]: z_t[{updated_state.evolution_step}] "
            f"confidence={updated_state.confidence:.2f}, "
            f"hypotheses={len(updated_state.hypotheses)}, "
            f"patterns={len(updated_state.learned_patterns)}"
        )

    # ------------------------------------------------------------------
    # Managed evolution helpers (NAT+)
    # ------------------------------------------------------------------

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
        window = metrics[-self.cfg.plateau_window:]
        if len(window) < self.cfg.plateau_window:
            return False
        best_so_far = window[0]
        for metric in window[1:]:
            if self.lower_is_better:
                if metric < best_so_far:
                    return False
            else:
                if metric > best_so_far:
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

    def _should_reflect(self, z_t: CognitiveState, feedback: Feedback):
        """Decide whether to spend an LLM call on reflect_op."""
        if not self.cfg.triggered_reflection:
            return self.cfg.reflect_after_every_step, "always"
        trigger = self._get_reflection_trigger(z_t, feedback)
        if trigger != "routine":
            return True, trigger
        return self.cfg.reflect_on_routine, trigger

    @staticmethod
    def _clamp_confidence(value: float) -> float:
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
        """Apply conservative decay so unhealthy branches recover faster."""
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

    # ------------------------------------------------------------------
    # Causal intervention
    # ------------------------------------------------------------------

    def _apply_intervention(self, z_after_reflect: CognitiveState, step: int) -> CognitiveState:
        """Apply causal intervention to z_t after reflect (Exp 3).

        Identical logic to MC-ESES _apply_intervention.
        """
        mode = self.cfg.intervention_mode
        if mode == "natural":
            return z_after_reflect

        if mode == "ablated":
            blank = CognitiveState()
            blank.environment_context = z_after_reflect.environment_context
            return blank

        if mode == "scrambled":
            if self._scrambled_state is not None:
                s = self._scrambled_state.clone()
                s.environment_context = z_after_reflect.environment_context
                return s
            blank = CognitiveState()
            blank.environment_context = z_after_reflect.environment_context
            return blank

        if mode == "frozen":
            if step <= self.cfg.intervention_freeze_step:
                if step == self.cfg.intervention_freeze_step:
                    self._frozen_state = z_after_reflect.clone()
                return z_after_reflect
            else:
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
        """Record trajectory snapshot for experiment analysis."""
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

        # Update tracking state
        if feedback.error_category:
            self._prev_error_categories.add(feedback.error_category)
        if feedback.metric is not None:
            self._recent_metrics.append(feedback.metric)
        if not feedback.is_buggy:
            self._has_any_valid = True

    def _run_environment_recon(self, state) -> str:
        """Probe execution environment (reuses MC-ESES recon logic)."""
        if "solver_interpreter" not in state:
            return ""
        try:
            from dojo.solvers.mceses.mceses import MCESES
            # Run the recon script in the interpreter
            exec_result = state["solver_interpreter"].run(
                MCESES._RECON_SCRIPT, include_exec_time=False,
            )
            raw = "\n".join(exec_result.term_out)
        except Exception as e:
            self.logger.warning(f"Environment recon failed ({e})")
            return ""

        # Parse JSON output (same logic as MC-ESES)
        start = raw.find("===RECON_JSON_START===")
        end = raw.find("===RECON_JSON_END===")
        if start < 0 or end < 0:
            return raw[:2000] if raw else ""

        json_str = raw[start + len("===RECON_JSON_START==="):end].strip()
        try:
            report = json.loads(json_str)
        except json.JSONDecodeError:
            return json_str[:2000]

        lines = []
        lines.append(f"Python {report.get('python', '?')}")
        pkg_keys = ["torch", "transformers", "scikit-learn", "lightgbm",
                     "xgboost", "datasets", "evaluate"]
        pkg_parts = [f"{k}=={report[k]}" for k in pkg_keys if k in report]
        if pkg_parts:
            lines.append(f"Key packages: {', '.join(pkg_parts)}")
        gpu_count = report.get("gpu_count", 0)
        if gpu_count:
            lines.append(f"GPU: {gpu_count}x {report.get('gpu_name', '?')} ({report.get('gpu_memory_gb', '?')} GB)")
        else:
            lines.append("GPU: none (CPU only)")
        lines.append(f"CPUs: {report.get('cpu_count', '?')}")
        for c in report.get("compatibility", []):
            lines.append(f"[OK] {c}")
        for w in report.get("warnings", []):
            lines.append(f"[WARNING] {w}")
        data_files = report.get("data_files", [])
        if data_files:
            lines.append(f"Data directory: {len(data_files)} files")
            for f in data_files[:10]:
                lines.append(f"  - {f}")

        summary = "\n".join(lines)
        self.logger.info(f"Environment recon complete:\n{summary}")
        return summary

    # ------------------------------------------------------------------
    # Package discovery & data preview
    # ------------------------------------------------------------------

    def discover_packages(self, state):
        """Run ``pip list`` inside the container to discover installed packages.

        Updates ``self.cfg.available_packages`` so every subsequent operator
        prompt reflects the real environment instead of the static default.
        """
        if "solver_interpreter" not in state:
            self.logger.warning("No interpreter available — using static package list")
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
            self.logger.warning(f"pip list discovery failed ({e}) — using static package list")
            return

        if not packages:
            self.logger.warning("pip list returned no packages — using static package list")
            return

        self.logger.info(f"Discovered {len(packages)} packages in container (was {len(self.cfg.available_packages)})")

        # Update config so all operators automatically see the real list.
        try:
            from omegaconf import OmegaConf

            OmegaConf.update(self.cfg, "available_packages", packages, force_add=True)
        except Exception:
            # OmegaConf may be frozen; fall back to direct attribute set.
            try:
                self.cfg.available_packages = packages
            except Exception as e2:
                self.logger.warning(f"Could not update available_packages on cfg ({e2})")

    def update_data_preview(self, state):
        """
        Generate a data preview to provide context for the LLM operators.

        Creates a small preview of the data (head, shapes, etc.) that can be used
        to help the LLM understand the data structure when generating solutions.

        Args:
            state: The current solver state containing the interpreter
        """
        assert "solver_interpreter" in state, (
            "For generating data previews, the solver needs access to an interpreter."
        )

        self.logger.debug("Generating data preview")
        if state["solver_interpreter"].local:
            self.data_preview = data_preview.generate(state["solver_interpreter"].working_dir)
        else:
            import inspect

            path = Path(inspect.getsourcefile(data_preview))
            script = path.read_text()
            code = f"{script}\nprint(generate(Path('.').resolve()))"
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            self.data_preview = "\n".join(exec_result.term_out)
        self.logger.debug("Data preview generated")

    def step(self, task, state):
        """
        Execute a single iteration of the Greedy solver process.

        This method implements the core Greedy algorithm:
        1. Select a node to work on (draft/debug/improve)
        2. Apply the appropriate operator to generate new code
        3. Evaluate the code using the task
        4. Parse the results and update the journal

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and evaluation results
        """
        self.logger.info(f"Step {self.state.current_step}: Starting iteration")

        # Possibly generate data preview first
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        # Select the parent node
        parent_node = self.search_policy()
        self.logger.debug(f"Step {self.state.current_step}: Selected parent node: {parent_node}")

        # If no parent node is selected, draft a new solution.
        # Otherwise, if the parent node is buggy, debug it.
        # Otherwise, improve the parent node.
        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        # Evaluate the code
        self.logger.debug(f"Step {self.state.current_step}: Executing generated code")
        state, eval_result = task.step_task(state, extract_code(result_node.code))

        # Update running time
        # self.state.running_time += eval_result[EXECUTION_OUTPUT].exec_time

        # Parse the evaluation results
        self.parse_eval_result(node=result_node, eval_result=eval_result)

        # Store in the journal
        self.journal.append(result_node)

        # --- Cognitive state evolution (if enabled) ---
        # NAT: reflect_after_every_step controls whether to reflect
        # NAT+: managed_evolution handles its own trigger logic internally
        if self.cfg.use_cognitive_state and (self.cfg.reflect_after_every_step or self.cfg.managed_evolution):
            self._reflect_and_record(result_node)

        # Log the best node
        best_node = self.journal.get_best_node()
        best_node_step = 0 if best_node is None else best_node.step

        # Log the latest node
        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

        # Log state
        self.logger.log(
            self.state.state_dict(),
            "STATE",
            step=self.state.current_step,
        )

        self.logger.info(f"Step {self.state.current_step}: Iteration complete")
        return state, eval_result

    def parse_eval_result(self, node: Node, eval_result: Dict[str, Any]):
        """
        Parse evaluation results and update the node accordingly.

        Processes the execution output, extracts metrics, determines if the solution
        is buggy, and updates the node with this information. Also applies the analysis
        operator to get additional insights about the solution.

        Args:
            node: The node to update with evaluation results
            eval_result: Dictionary containing evaluation results from task execution
        """
        self.logger.debug(f"Parsing execution results for node {node.id}")

        # Safely ensure we have eval_result
        if isinstance(eval_result, dict):
            assert EXECUTION_OUTPUT in eval_result
        else:
            raise ValueError(f"Unexpected eval_result type: {type(eval_result)}")

        # Absorb the execution output into the node
        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])

        # Safely perform the analyze operation
        try:
            response = self._analyze(node)
        except Exception as e:
            self.logger.error(f"Error during analysis operator: {str(e)}")
            response = {}

        # Parse response to dictionary
        # If the response is a string, we try to parse it into a dictionary
        response = parse_json_output(response)

        # Validate it's actually a dictionary
        if not isinstance(response, dict):
            self.logger.warning(f"Parsed response is not a dictionary: {type(response)}")
            response = {}

        # If the response is empty, we return a default dictionary
        if len(response) == 0:
            response = {"metric": None, "summary": "", "is_bug": True}
        else:
            if "metric" not in response:
                response["metric"] = None
            if "summary" not in response:
                response["summary"] = ""
            if "is_bug" not in response:
                if response["metric"] is not None:
                    response["is_bug"] = False
                else:
                    response["is_bug"] = True

        # If the metric isn't a float or int then fill the metric with the worst metric
        if not isinstance(response["metric"], (float, int)):
            response["metric"] = None

        # If a validation fitness value is provided (not test fitness) from the task
        # we replace the validation metric with it.
        if eval_result.get(VALIDATION_FITNESS, None) is not None:
            response["metric"] = float(eval_result[VALIDATION_FITNESS])

        # Store the analysis summary
        node.analysis = response["summary"]

        # Extract potential auxliary evaluation information to store
        # in the node. This is more for logging purposes.
        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {})
        if self.cfg.use_test_score:
            test_score = aux_eval_info.get("score", None)
            aux_eval_info["validation_score"] = response["metric"]
            response["metric"] = test_score
            self.logger.info(f"Using test score: {test_score}")

        # Determine if solution is valid
        # If the task does not return this key, we assume the solution is valid
        valid_solution = eval_result.get(VALID_SOLUTION, True)
        validity_feedback = eval_result.get(VALID_SOLUTION_FEEDBACK, None)
        if validity_feedback is not None:
            aux_eval_info["validity_feedback"] = validity_feedback
            validity_feedback = f"\n\n submission.csv Grader Feedback: {validity_feedback}"
            node._term_out.append(validity_feedback)
        else:
            aux_eval_info["validity_feedback"] = "submission grader feedback not available"

        node.is_buggy = (
            response["is_bug"] or (not node.exit_code == 0) or (response["metric"] is None) or (not valid_solution)
        )

        if node.is_buggy:
            node.metric = WorstMetricValue(info=aux_eval_info)
            self.logger.debug(f"Node {node.id} marked as buggy")
        else:
            node.metric = MetricValue(response["metric"], maximize=not self.lower_is_better, info=aux_eval_info)
            self.logger.debug(f"Node {node.id} metric: {response['metric']}")
