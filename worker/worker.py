"""
GPU Pod - Windows GPU Worker
=============================
Minimal worker. Registers with server, heartbeats, runs GPU compute jobs.

Setup (Windows with NVIDIA GPU):
    uv venv --python 3.13
    .venv\Scripts\activate
    uv sync
    set GPU_POD_SERVER_URL=http://192.168.1.42:8000
    python worker.py

Run:  python worker.py
"""

import os, time, json, sys, traceback, subprocess
from pathlib import Path

import requests

CONFIG = {
    "server_url": os.environ.get("GPU_POD_SERVER_URL", "http://localhost:8000"),
    "worker_name": os.environ.get("GPU_POD_WORKER_NAME", "windows-gtx1650"),
    "heartbeat_interval": 5,
}

WORKER_ID = None
WORK_DIR = Path("work")
WORK_DIR.mkdir(exist_ok=True)


# ─── GPU Detection ───

def detect_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            shell=True, text=True, timeout=10,
        )
        parts = [p.strip() for p in out.strip().split("\n")[0].split(",")]
        vram_gb = round(float(parts[1]) / 1024, 1) if len(parts) > 1 else 0
        return parts[0], vram_gb
    except Exception:
        return "unknown", 0.0


# ─── Server Comms ───

def register():
    global WORKER_ID
    gpu_model, vram_gb = detect_gpu()
    try:
        r = requests.post(f"{CONFIG['server_url']}/register-worker", json={
            "name": CONFIG["worker_name"], "gpu_model": gpu_model, "vram_gb": vram_gb,
        }, timeout=10)
        r.raise_for_status()
        WORKER_ID = r.json()["worker_id"]
        print(f"[OK] Registered | ID: {WORKER_ID} | GPU: {gpu_model} | VRAM: {vram_gb}GB")
        return True
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot reach server at {CONFIG['server_url']}")
        return False
    except Exception as e:
        print(f"[FAIL] Registration error: {e}")
        return False

def send_heartbeat():
    try:
        r = requests.post(f"{CONFIG['server_url']}/heartbeat", json={
            "worker_id": WORKER_ID, "status": "idle",
        }, timeout=10)
        r.raise_for_status()
        return r.json().get("assigned_job")
    except Exception as e:
        print(f"[WARN] Heartbeat failed: {e}")
        return None

def send_result(job_id, success, error=""):
    try:
        requests.post(f"{CONFIG['server_url']}/result", json={
            "worker_id": WORKER_ID, "job_id": job_id, "success": success, "error": error,
        }, timeout=10)
    except Exception as e:
        print(f"[FAIL] Could not send result: {e}")


# ─── Job Execution — Simple GPU Compute ───

def execute_job(job: dict):
    job_id = job["job_id"]
    task_type = job["task_type"]
    prompt = job.get("prompt", "")
    params = job.get("params", {})

    print(f"\n{'='*50}")
    print(f"[JOB] {job_id} | Type: {task_type}")
    if prompt:
        print(f"      Prompt: {prompt[:80]}")
    print(f"{'='*50}")

    try:
        import torch

        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this machine!")

        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] {gpu_name}")

        result = {}

        if task_type == "matrix-multiply" or task_type == "compute":
            # Matrix multiplication benchmark
            size = params.get("size", 2048)
            print(f"[WORK] Multiplying {size}x{size} matrices on GPU...")
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            # Warmup
            c = a @ b
            torch.cuda.synchronize()

            # Timed run
            start = time.time()
            c = a @ b
            torch.cuda.synchronize()
            elapsed = time.time() - start

            flops = 2 * size**3 / elapsed / 1e12
            result = {
                "operation": f"{size}x{size} matrix multiply",
                "time_seconds": round(elapsed, 4),
                "tflops": round(flops, 2),
                "device": gpu_name,
            }
            print(f"[DONE] {result}")

        elif task_type == "tensor-info":
            # Report GPU tensor info
            result = {
                "device": gpu_name,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            }
            print(f"[DONE] GPU info: {result}")

        else:
            # Generic "run this python code on the GPU"
            code = prompt or params.get("code", "")
            if not code:
                raise ValueError(f"Unknown task '{task_type}'. Try: compute, tensor-info, or send code in 'prompt' or params.code")

            print(f"[WORK] Executing user code...")
            local_vars = {"torch": torch, "device": torch.device("cuda"),
                          "WORK_DIR": str(WORK_DIR), "job_id": job_id, "params": params}
            exec(code, local_vars)
            result = local_vars.get("result", "Code executed successfully")
            print(f"[DONE] Code executed")

        # Save result
        out_path = WORK_DIR / f"{job_id}_result.json"
        out_path.write_text(json.dumps(result, indent=2))
        send_result(job_id, success=True)

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        send_result(job_id, success=False, error=error_msg)


# ─── Main Loop ───

def main():
    print("=" * 50)
    print("  GPU Pod Worker")
    print("=" * 50)

    if not register():
        time.sleep(10)
        if not register():
            sys.exit(1)

    busy = False
    while True:
        if not busy:
            job = send_heartbeat()
            if job:
                busy = True
                execute_job(job)
                busy = False
        time.sleep(CONFIG["heartbeat_interval"])

if __name__ == "__main__":
    main()
