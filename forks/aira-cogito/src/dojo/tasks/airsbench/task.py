import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from dojo.config_dataclasses.task.airsbench import AIRSBenchTaskConfig
from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
from dojo.utils.code_parsing import extract_code


class AIRSBenchTask(Task):
    _solution_script = "solution.py"
    _submission_file_path = None

    def __init__(self, cfg: AIRSBenchTaskConfig) -> None:
        super().__init__(cfg)

        self.task_dir = Path(cfg.task_dir).resolve()
        self.prepare_script = self.task_dir / "prepare.py"
        self.evaluate_prepare_script = self.task_dir / "evaluate_prepare.py"
        self.evaluate_script = self.task_dir / "evaluate.py"
        self.project_description_path = self.task_dir / "project_description.md"
        self.metadata_path = self.task_dir / "metadata.yaml"
        self.prepared_data_dir = Path(cfg.prepared_data_dir).resolve()
        self.evaluation_data_dir = Path(cfg.evaluation_data_dir).resolve()
        self.results_output_dir = Path(cfg.results_output_dir).resolve()
        self.global_shared_data_dir = Path(cfg.global_shared_data_dir).resolve()

        self.prepared_data_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        self.task_description = self.project_description_path.read_text()
        self.metric_name = self.metadata["logging_info"]["metric"]
        self.lower_is_better = self.metadata["metric_lower_is_better"]

    def _reset_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _run_script(self, script_path: Path, args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=str(cwd or self.task_dir),
            capture_output=True,
            text=True,
            check=True,
        )

    def _parse_eval_stdout(self, stdout: str) -> Dict[str, Any]:
        pattern = re.compile(r"--- EVALUATION RESULT ---\s*(\{[\s\S]*?\})", re.DOTALL)
        match = pattern.search(stdout)
        if not match:
            raise ValueError(f"Failed to extract evaluation JSON from stdout:\n{stdout}")
        return json.loads(match.group(1))

    def _evaluate_submission(self) -> Tuple[float, Dict[str, Any]]:
        self._reset_dir(self.evaluation_data_dir)
        self._run_script(
            self.evaluate_prepare_script,
            [
                "--global-shared-data-dir",
                str(self.global_shared_data_dir),
                "--agent-data-mount-dir",
                str(self.evaluation_data_dir),
                "--agent-log-dir",
                str(self._submission_file_path.parent),
            ],
        )

        with tempfile.TemporaryDirectory(prefix="airsbench_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "data").symlink_to(self.evaluation_data_dir)
            (tmp_path / "logs").symlink_to(self._submission_file_path.parent)
            result = self._run_script(
                self.evaluate_script,
                ["--submission-file", "./data/submission.csv"],
                cwd=tmp_path,
            )

        report = self._parse_eval_stdout(result.stdout)
        score = report.get(self.metric_name)
        if score is None:
            numeric_values = [v for v in report.values() if isinstance(v, (int, float))]
            if len(numeric_values) != 1:
                raise ValueError(f"Could not determine primary metric from report: {report}")
            score = numeric_values[0]

        report["metric_lower_is_better"] = self.lower_is_better
        return float(score), report

    def prepare(self, **task_args):
        self._reset_dir(self.prepared_data_dir)
        state = task_args
        state["init_obs"] = {}

        self._run_script(
            self.prepare_script,
            [
                "--global-shared-data-dir",
                str(self.global_shared_data_dir),
                "--agent-data-mount-dir",
                str(self.prepared_data_dir),
                "--agent-log-dir",
                str(self.results_output_dir),
            ],
        )

        self._submission_file_path = Path(task_args["solver_interpreter"].working_dir) / self.cfg.submission_fname

        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": self.lower_is_better,
        }
        return state, task_info

    def step_task(self, state: Dict[str, Any], action: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            solution = extract_code(action)
        except Exception as e:
            exec_output = ExecutionResult.get_empty()
            exec_output.term_out[0] = f"Invalid solution: {e}"
            return state, {EXECUTION_OUTPUT: exec_output, VALIDATION_FITNESS: None, VALID_SOLUTION: False}

        interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(solution, file_name=self._solution_script)
        eval_result = {
            EXECUTION_OUTPUT: exec_output,
            VALID_SOLUTION: False,
        }

        if (not exec_output.exit_code == 0) or exec_output.timed_out:
            self._submission_file_path.unlink(missing_ok=True)
            return state, eval_result

        interpreter.fetch_file(self._submission_file_path)
        if not self._submission_file_path.exists():
            eval_result[VALID_SOLUTION_FEEDBACK] = f"Missing {self.cfg.submission_fname}"
            return state, eval_result

        # Keep a copy of the latest submission for debugging / reproducibility.
        try:
            archive_dir = self.results_output_dir / "submissions"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._submission_file_path, archive_dir / self.cfg.submission_fname)
        except Exception:
            pass

        try:
            test_fitness, report = self._evaluate_submission()
            eval_result[VALID_SOLUTION] = True
            eval_result[VALID_SOLUTION_FEEDBACK] = "Submission evaluated successfully."

            # Make the metric visible to solvers that only look at VALIDATION_FITNESS,
            # and normalize the primary key expected elsewhere in the stack.
            eval_result[TEST_FITNESS] = test_fitness
            eval_result[VALIDATION_FITNESS] = test_fitness
            report["score"] = float(test_fitness)
            report["metric_name"] = self.metric_name
            eval_result[AUX_EVAL_INFO] = report
        except Exception as e:
            eval_result[VALID_SOLUTION_FEEDBACK] = str(e)
        finally:
            # Keep the archived copy under results_output_dir/submissions/.
            self._submission_file_path.unlink(missing_ok=True)

        if interpreter.factory:
            interpreter.close()
        else:
            interpreter.run(f"!rm -f {self.cfg.submission_fname}")

        return state, eval_result

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._submission_file_path is None:
            raise ValueError("The path to the submission file must be set.")

        exec_output = interpreter.run(solution, file_name=self._solution_script)
        eval_result = {EXECUTION_OUTPUT: exec_output}

        interpreter.fetch_file(self._submission_file_path)
        assert self._submission_file_path.exists(), "The final solution is not valid."

        test_fitness, report = self._evaluate_submission()
        eval_result[TEST_FITNESS] = test_fitness
        eval_result[VALIDATION_FITNESS] = test_fitness
        report["score"] = float(test_fitness)
        report["metric_name"] = self.metric_name
        eval_result[AUX_EVAL_INFO] = report
        return eval_result

    def close(self, state):
        for interp_key in ["solver_interpreter", "eval_interpreter"]:
            if interp_key not in state:
                continue

            interpreter = state[interp_key]
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()

            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
