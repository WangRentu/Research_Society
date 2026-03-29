from dataclasses import dataclass, field

from omegaconf import SI

from dojo.config_dataclasses.task.base import TaskConfig


@dataclass
class AIRSBenchTaskConfig(TaskConfig):
    benchmark: str = field(
        default="airsbench",
        metadata={
            "help": "Type of the task.",
        },
    )

    data_dir: str = field(
        default=SI("${task.prepared_data_dir}"),
        metadata={
            "help": "Directory exposed to the agent as ./data.",
            "exclude_from_hash": True,
        },
    )
    tasks_root_dir: str = field(
        default=SI("${oc.env:AIRS_BENCH_TASKS_DIR}"),
        metadata={
            "help": "Root directory containing AIRS-Bench RAD task folders.",
            "exclude_from_hash": True,
        },
    )
    global_shared_data_dir: str = field(
        default=SI("${oc.env:AIRS_BENCH_DATA_DIR}"),
        metadata={
            "help": "Root directory containing AIRS-Bench raw datasets.",
            "exclude_from_hash": True,
        },
    )
    task_dir: str = field(
        default=SI("${task.tasks_root_dir}/${task.name}"),
        metadata={
            "help": "Absolute path to the AIRS-Bench RAD task folder.",
            "exclude_from_hash": True,
        },
    )
    prepared_data_dir: str = field(
        default=SI("${logger.output_dir}/task_data/public"),
        metadata={
            "help": "Prepared public data directory for the agent.",
            "exclude_from_hash": True,
        },
    )
    evaluation_data_dir: str = field(
        default=SI("${logger.output_dir}/task_data/eval"),
        metadata={
            "help": "Temporary evaluation data directory.",
            "exclude_from_hash": True,
        },
    )
    submission_fname: str = field(
        default="submission.csv",
        metadata={
            "help": "Submission file expected from the agent.",
            "exclude_from_hash": True,
        },
    )
    results_output_dir: str = field(
        default=SI("${logger.output_dir}/results"),
        metadata={
            "help": "Directory where evaluation artifacts are stored.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
