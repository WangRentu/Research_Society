# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ENV_VAR_NOT_FOUND_ERR = (
    "{env_var_name} environment variable is not set or is empty."
    " Make sure to set it in your .env file or in the environment variables."
)


def _read_int_env(name: str) -> int | None:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def get_num_cpus() -> int:
    """Return the number of CPUs available to this job.

    Prefers SLURM allocation when present.
    """
    return (
        _read_int_env("SLURM_CPUS_PER_TASK")
        or _read_int_env("SLURM_CPUS_ON_NODE")
        or (os.cpu_count() or 1)
    )


def get_cuda_visible_devices() -> str | None:
    """Return CUDA_VISIBLE_DEVICES as a string if set, else None."""
    v = os.getenv("CUDA_VISIBLE_DEVICES")
    if v is None or v.strip() == "":
        return None
    return v.strip()


def _count_visible_gpus_from_env() -> int | None:
    v = get_cuda_visible_devices()
    if v is None:
        return None
    # Common forms: "0,1,2,3" or "0" or "0, 1"
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    return len(parts)


def get_gpu_overview() -> tuple[int, str]:
    """Return (num_gpus, gpu_description).

    Uses nvidia-smi if available; gracefully falls back to CPU-only.
    """
    visible_count = _count_visible_gpus_from_env()

    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
            shell=True,
            text=True,
        )
        lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
        if not lines:
            return 0, "CPU-only"

        # lines like: "NVIDIA H200, 141312" (MiB)
        names = []
        mem_gb = []
        for ln in lines:
            if "," in ln:
                name, mem = ln.split(",", 1)
                names.append(name.strip())
                try:
                    mem_mib = float(mem.strip())
                    mem_gb.append(mem_mib / 1024.0)
                except ValueError:
                    pass
            else:
                names.append(ln)

        # Prefer the visible count if CUDA_VISIBLE_DEVICES is set; otherwise use nvidia-smi count.
        n = visible_count if visible_count is not None else len(lines)

        # If multiple models, just list unique names.
        uniq_names = sorted(set(names))
        if mem_gb:
            # Report min/max across GPUs; keeps it compact.
            mem_min = min(mem_gb)
            mem_max = max(mem_gb)
            mem_part = (
                f"~{mem_min:.1f}-{mem_max:.1f} GB" if mem_min != mem_max else f"~{mem_min:.1f} GB"
            )
            desc = f"{', '.join(uniq_names)} ({mem_part} per GPU)"
        else:
            desc = ", ".join(uniq_names)

        return n, desc

    except subprocess.CalledProcessError:
        return 0, "CPU-only"


def get_ram_gb() -> float | None:
    """Best-effort total RAM in GB, without extra dependencies."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # kB
                    kb = float(parts[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def get_compute_overview() -> str:
    """Short, Markdown-friendly compute summary injected into prompts."""
    n_cpus = get_num_cpus()
    n_gpus, gpu_desc = get_gpu_overview()
    visible = get_cuda_visible_devices()
    ram_gb = get_ram_gb()

    lines = []
    if n_gpus > 0:
        lines.append(f"- GPUs: {n_gpus} × {gpu_desc}")
        if visible is not None:
            lines.append(f"- CUDA_VISIBLE_DEVICES: {visible}")
    else:
        lines.append("- GPUs: 0 (CPU-only)")
        if visible is not None:
            lines.append(f"- CUDA_VISIBLE_DEVICES: {visible} (no GPUs detected)")

    lines.append(f"- CPUs: {n_cpus} logical cores")
    if ram_gb is not None:
        lines.append(f"- RAM: ~{ram_gb:.0f} GB")

    lines.append(
        "- Guidance: use all visible GPUs when using torch; enable AMP when appropriate; "
        "for CPU ML set n_jobs/num_threads≈CPUs and avoid nested parallelism."
    )

    return "\n".join(lines)


def get_packages_from_requirements(requirements_path: str | None = None) -> list[str]:
    """Parse pip.requirements.txt and return a deduplicated list of package names.

    Tries common locations if *requirements_path* is not given.  Returns an
    empty list when the file cannot be found.
    """
    if requirements_path is None:
        candidates = [
            Path(__file__).resolve().parents[3] / "superimage" / "pip.requirements.txt",
            Path(os.getenv("SUPERIMAGE_DIR", "")) / "pip.requirements.txt",
        ]
        for c in candidates:
            if c.is_file():
                requirements_path = str(c)
                break

    if requirements_path is None or not Path(requirements_path).is_file():
        return []

    packages: list[str] = []
    with open(requirements_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # skip git+/http URLs and option flags
            if line.startswith(("git+", "http", "-")):
                continue
            # strip version specifiers and extras
            name = re.split(r"[>=<!~\[;]", line)[0].strip()
            if name:
                packages.append(name)

    # deduplicate, preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in packages:
        key = p.lower().replace("-", "_")
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def parse_pip_list_output(pip_output: str) -> list[str]:
    """Parse the stdout of ``pip list --format=freeze`` into package names."""
    packages: list[str] = []
    for line in pip_output.splitlines():
        line = line.strip()
        if "==" in line:
            name = line.split("==")[0].strip()
            if name:
                packages.append(name)
    return packages


def get_hardware():
    """Determine available hardware (GPU or CPU)."""
    try:
        # Check if `nvidia-smi` is available and get GPU name
        result = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True)
        # Process output: trim spaces, remove duplicates, and format
        hardware = ", ".join(sorted(set(line.strip() for line in result.split("\n") if line.strip())))
    except subprocess.CalledProcessError:
        hardware = "a CPU"  # Default if no GPU is found
    return hardware


def check_pytorch_gpu():
    """Check if PyTorch can use a GPU."""
    try:
        return subprocess.check_output(
            "python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "ERROR: PyTorch check failed"


def check_tensorflow_gpu():
    """Check if TensorFlow can use a GPU."""
    try:
        return subprocess.check_output(
            "python -c \"import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "ERROR: TensorFlow check failed"


def format_time(seconds):
    """Convert time in seconds to a human-readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}hrs {minutes}mins {secs}secs"


def get_log_dir():
    """Get the log directory, creating it if it doesn't exist."""
    log_dir = os.getenv("LOGGING_DIR", "")
    if not log_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="LOGGING_DIR"))
    return log_dir


def get_superimage_dir():
    """Get the superimage directory, creating it if it doesn't exist."""
    superimage_dir = os.getenv("SUPERIMAGE_DIR", "")
    # raise an error if empty string or None
    if not superimage_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="SUPERIMAGE_DIR"))
    return superimage_dir


def get_mlebench_data_dir():
    """Get the MLEBench data directory, creating it if it doesn't exist."""
    mlebench_data_dir = os.getenv("MLE_BENCH_DATA_DIR", "")
    if not mlebench_data_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="MLE_BENCH_DATA_DIR"))
    return mlebench_data_dir


def get_default_slurm_partition():
    """Get the default Slurm partition from the configuration."""
    slurm_partition = os.getenv("DEFAULT_SLURM_PARTITION","")
    if not slurm_partition:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_PARTITION"))
    return slurm_partition


def get_default_slurm_account():
    """Get the default Slurm account from the configuration."""
    slurm_account = os.getenv("DEFAULT_SLURM_ACCOUNT", "")
    if not slurm_account:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_ACCOUNT"))
    return slurm_account


def get_default_slurm_qos():
    """Get the default Slurm QoS from the configuration."""
    slurm_qos = os.getenv("DEFAULT_SLURM_QOS", "")
    if not slurm_qos:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_QOS"))
    return slurm_qos
