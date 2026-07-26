"""
GPU detection and stats via nvidia-smi and PyTorch.
"""

import subprocess
from pathlib import Path

import requests

WORK_DIR = Path.home() / ".gpu-pod-work"
WORK_DIR.mkdir(exist_ok=True)


def get_gpu_stats_nvidia():
    """Get live GPU stats from nvidia-smi. Returns dict or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            shell=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in out.strip().split("\n")[0].split(",")]
        return {
            "name": parts[0] if len(parts) > 0 else "Unknown",
            "util": float(parts[1]) if len(parts) > 1 else 0,
            "mem_used": float(parts[2]) if len(parts) > 2 else 0,
            "mem_total": float(parts[3]) if len(parts) > 3 else 0,
            "temp": float(parts[4]) if len(parts) > 4 else 0,
            "power": parts[5] if len(parts) > 5 else "N/A",
        }
    except Exception:
        return None


def get_gpu_stats_torch():
    """Get GPU stats from PyTorch."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "name": torch.cuda.get_device_name(0),
            "util": 0,
            "mem_used": round(torch.cuda.memory_allocated(0) / 1024**3, 1) * 1024,
            "mem_total": round(props.total_memory / 1024**3, 1) * 1024,
            "temp": 0,
            "power": "N/A",
        }
    except Exception:
        return None


def check_torch_cuda():
    """Returns True if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
