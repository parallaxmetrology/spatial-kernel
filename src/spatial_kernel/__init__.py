"""Spatial Kernel: deterministic, task-relative spatial verification of robot outcomes (research name: VTL spatial compliance)."""

from .core import KernelConfig, compute_kernel_metrics_from_bgr, compute_kernel_metrics_from_path
from .isolate import IsolationResult, isolate_object
from .parallax import ParallaxVerdict, Read, fuse_reads
from .tasks import TaskProfile, get_task_profile
from .verify import verify_parallax, verify_spatial_task

__all__ = [
    "KernelConfig",
    "TaskProfile",
    "IsolationResult",
    "ParallaxVerdict",
    "Read",
    "compute_kernel_metrics_from_bgr",
    "compute_kernel_metrics_from_path",
    "fuse_reads",
    "get_task_profile",
    "isolate_object",
    "verify_parallax",
    "verify_spatial_task",
]
