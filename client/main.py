"""
GPU Pod Desktop
===============
Entry point. Run with:  uv run python main.py
"""

import sys
from pathlib import Path

# Make root gpu_pod_otel importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_pod_otel import init_otel, shutdown as otel_shutdown

# Init OTel for client (User mode) — no PyTorch needed
init_otel("gpu-pod-client", env="development")

from client.app import GPUApp

if __name__ == "__main__":
    try:
        app = GPUApp()
        app.run()
    finally:
        otel_shutdown()
