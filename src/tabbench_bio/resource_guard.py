"""Check the reference execution environment before benchmark work starts."""

import os
import sys


class HardwareMismatchError(RuntimeError):
    """The execution environment does not match the reference resource budget."""


def check_reference_hardware(
    *, gpu_required=False, gpu_name=None, model_threads=None, strict=None, probe_gpu=True
):
    """Warn by default; TABBENCH_STRICT_HARDWARE=1 aborts on mismatches.

    This checks observations and never invents allocation metadata in results.
    CPU-only runs do not require a GPU allocation. Model/library thread counts
    are independent of the allocated CPU budget.
    """
    issues = []

    def integer(name):
        value = os.environ.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            issues.append(f"{name}={value!r} is not an integer")
            return None

    cpus = integer("SLURM_CPUS_PER_TASK")
    memory = integer("SLURM_MEM_PER_NODE")
    if memory is None and os.environ.get("SLURM_MEM_PER_CPU"):
        per_cpu = integer("SLURM_MEM_PER_CPU")
        if per_cpu is not None and cpus is not None:
            memory = per_cpu * cpus
    if cpus is None:
        issues.append("CPU allocation cannot be verified without SLURM_CPUS_PER_TASK")
    elif cpus != 16:
        issues.append(f"allocated CPUs={cpus}, expected 16")
    if memory is None:
        issues.append("system-RAM allocation cannot be verified from Slurm metadata")
    elif memory != 81920:
        issues.append(f"allocated system RAM={memory} MiB, expected 81920 MiB (80 GiB)")
    if gpu_required:
        if gpu_name is None and probe_gpu:
            try:
                import torch

                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            except (ImportError, AttributeError, RuntimeError):
                gpu_name = None
        if gpu_name is None or "L40S" not in gpu_name.upper():
            issues.append(f"GPU={gpu_name or 'unavailable'}, expected NVIDIA L40S")
    if issues:
        message = "Non-reference benchmark hardware: " + "; ".join(issues)
        if strict is None:
            strict = os.environ.get("TABBENCH_STRICT_HARDWARE", "").lower() in {"1", "true", "yes"}
        if strict:
            raise HardwareMismatchError(message)
        print("WARNING: " + message, file=sys.stderr, flush=True)
    return {"allocated_cpus": cpus, "allocated_memory_mb": memory, "gpu": gpu_name}
