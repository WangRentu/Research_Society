# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def _build_mlebench_task(cfg, **kwargs):
    from dojo.tasks.mlebench.task import MLEBenchTask

    return MLEBenchTask(cfg, **kwargs)


def _build_airsbench_task(cfg, **kwargs):
    from dojo.tasks.airsbench.task import AIRSBenchTask

    return AIRSBenchTask(cfg, **kwargs)


TASK_MAP = {
    "MLEBenchTaskConfig": _build_mlebench_task,
    "AIRSBenchTaskConfig": _build_airsbench_task,
}
