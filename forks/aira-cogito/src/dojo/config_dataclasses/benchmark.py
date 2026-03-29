# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from typing import Any
from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig

from dojo.utils.config import build
from dojo.config_dataclasses.task import TASK_MAP
from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig
from dojo.config_dataclasses.task.airsbench import AIRSBenchTaskConfig


@dataclass
class BenchmarkConfig(BaseConfig):
    name: str = field(
        default=MISSING,
        metadata={
            "help": "Benchmark name, e.g., 'mlebench', 'AIRS-H'.",
        },
    )

    tasks: list[str] = field(
        default=MISSING,
        metadata={
            "help": "List of task name, e.g., ['random-acts-of-pizza'].",
        },
    )

    overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Overrides for all tasks.",
        },
    )

    task_overrides: dict[str, dict] = field(
        default_factory=dict,
        metadata={
            "help": "Overrides for specific tasks, allowing customization of task configurations.",
        },
    )

    def validate(self) -> None:
        super().validate()

    def to_cfg_list(self):
        """Convert benchmark configuration into a list of TaskConfig objects."""
        task_cfgs = []
        for task_name in self.tasks:
            overrides = {
                **self.overrides,
                **(self.task_overrides[task_name] if task_name in self.task_overrides else {}),
            }

            if self.name == "mlebench":
                task_cfg = MLEBenchTaskConfig(name=task_name, benchmark=self.name, **overrides)
            elif self.name == "airsbench":
                task_cfg = AIRSBenchTaskConfig(name=task_name, benchmark=self.name, **overrides)
            else:
                raise ValueError(
                    f"Unsupported benchmark '{self.name}'. Expected one of: ['mlebench', 'airsbench']."
                )

            task_cfgs.append(task_cfg)
        return task_cfgs
