"""
GPU Worker — runs in a background thread.
Registers with server, heartbeats, picks up jobs, executes GPU compute, returns results.
"""

import os
import time
import json
import threading
from pathlib import Path

import requests

from .gpu_utils import get_gpu_stats_nvidia, get_gpu_stats_torch, WORK_DIR


WORKER_ID = None
WORKER_RUNNING = False
SERVER_URL = None


def start(server_url: str, log_callback, stats_callback):
    """Start the worker in a background thread."""
    global SERVER_URL, WORKER_RUNNING, WORKER_ID
    SERVER_URL = server_url
    WORKER_RUNNING = True
    thread = threading.Thread(target=_loop, args=(log_callback, stats_callback), daemon=True)
    thread.start()
    return thread


def stop():
    """Signal the worker to stop."""
    global WORKER_RUNNING
    WORKER_RUNNING = False


def get_worker_id():
    global WORKER_ID
    return WORKER_ID


def _loop(log_callback, stats_callback):
    """Main worker loop running in background thread."""
    global WORKER_ID, WORKER_RUNNING

    gpu_stats = get_gpu_stats_nvidia() or get_gpu_stats_torch() or {}
    gpu_model = gpu_stats.get("name", "unknown")
    vram_total = gpu_stats.get("mem_total", 0)
    vram_gb = round(vram_total / 1024, 1) if vram_total else 0

    try:
        r = requests.post(f"{SERVER_URL}/register-worker", json={
            "name": os.environ.get("GPU_POD_WORKER_NAME", "gpu-worker"),
            "gpu_model": gpu_model,
            "vram_gb": vram_gb,
        }, timeout=10)
        r.raise_for_status()
        WORKER_ID = r.json()["worker_id"]
        log_callback(f"[OK] Registered | ID: {WORKER_ID} | GPU: {gpu_model}")
    except Exception as e:
        log_callback(f"[FAIL] Registration: {e}")
        WORKER_RUNNING = False
        return

    busy = False
    while WORKER_RUNNING:
        try:
            payload = {"worker_id": WORKER_ID, "status": "idle"}
            r = requests.post(f"{SERVER_URL}/heartbeat", json=payload, timeout=10)
            r.raise_for_status()

            stats = get_gpu_stats_nvidia()
            if stats:
                stats_callback(stats)

            job = r.json().get("assigned_job")
            if job:
                busy = True
                log_callback(f"[JOB] {job['job_id']} | Type: {job['task_type']}")
                _execute_job(job, log_callback)
                busy = False
        except Exception as e:
            if busy:
                log_callback(f"[WARN] Heartbeat failed: {e}")
        time.sleep(5)

    log_callback("[STOPPED] Worker terminated.")


def _execute_job(job: dict, log_callback):
    """Run a compute job on the GPU."""
    global WORKER_ID, SERVER_URL
    job_id = job["job_id"]
    task_type = job["task_type"]
    prompt = job.get("prompt", "")
    params = job.get("params", {})

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        log_callback(f"[GPU] {gpu_name}")

        result = {}

        if task_type in ("compute", "matrix-multiply"):
            size = params.get("size", 2048)
            log_callback(f"[WORK] {size}x{size} matrix multiply...")
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = a @ b
            torch.cuda.synchronize()
            start = time.time()
            c = a @ b
            torch.cuda.synchronize()
            elapsed = time.time() - start
            flops = 2 * size**3 / elapsed / 1e12
            result = {
                "operation": f"{size}x{size} matmul",
                "time_seconds": round(elapsed, 4),
                "tflops": round(flops, 2),
                "device": gpu_name,
            }
            log_callback(f"[DONE] {result['tflops']} TFLOPS")

        elif task_type == "tensor-info":
            result = {
                "device": gpu_name,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            }
            log_callback(f"[DONE] GPU info collected")

        out_path = WORK_DIR / f"{job_id}_result.json"
        out_path.write_text(json.dumps(result, indent=2))
        requests.post(f"{SERVER_URL}/result", json={
            "worker_id": WORKER_ID, "job_id": job_id, "success": True,
        }, timeout=10)
        log_callback(f"[OK] Result sent for {job_id}")

    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        log_callback(f"[ERROR] {e}")
        try:
            requests.post(f"{SERVER_URL}/result", json={
                "worker_id": WORKER_ID, "job_id": job_id, "success": False, "error": error_msg,
            }, timeout=10)
        except Exception:
            pass
