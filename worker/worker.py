"""
GPU Pod - Windows GPU Worker
=============================
Minimal worker with OpenTelemetry tracing → SigNoz.

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

# ── OTel ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_pod_otel import init_otel, tracer, logger, meter
from opentelemetry.trace import Status, StatusCode

__version__ = "0.1.0"

CONFIG = {
    "server_url": os.environ.get("GPU_POD_SERVER_URL", "http://localhost:8000"),
    "worker_name": os.environ.get("GPU_POD_WORKER_NAME", "windows-gtx1650"),
    "heartbeat_interval": 5,
}

WORKER_ID = None
WORK_DIR = Path("work")
WORK_DIR.mkdir(exist_ok=True)

# OTel metrics
jobs_executed = meter.create_counter("worker.jobs.executed", unit="1", description="Jobs executed by this worker")
job_duration_histogram = meter.create_histogram("worker.job.duration", unit="s", description="Job execution time")


def _log(msg: str):
    """Print to terminal AND send to SigNoz via OTel logger."""
    print(msg)
    # Strip ANSI/formatting for clean log body
    clean = msg.strip().lstrip("\n=")
    if clean:
        logger.info(clean)


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
    with tracer.start_as_current_span("worker.register") as span:
        gpu_model, vram_gb = detect_gpu()
        span.set_attribute("worker.gpu", gpu_model)
        span.set_attribute("worker.vram_gb", vram_gb)
        try:
            r = requests.post(f"{CONFIG['server_url']}/register-worker", json={
                "name": CONFIG["worker_name"], "gpu_model": gpu_model, "vram_gb": vram_gb,
            }, timeout=10)
            r.raise_for_status()
            WORKER_ID = r.json()["worker_id"]
            span.set_attribute("worker.id", WORKER_ID)
            _log(f"[OK] Registered | ID: {WORKER_ID} | GPU: {gpu_model} | VRAM: {vram_gb}GB")
            return True
        except requests.exceptions.ConnectionError:
            _log(f"[FAIL] Cannot reach server at {CONFIG['server_url']}")
            return False
        except Exception as e:
            _log(f"[FAIL] Registration error: {e}")
            return False

def send_heartbeat():
    with tracer.start_as_current_span("worker.heartbeat") as span:
        span.set_attribute("worker.id", WORKER_ID or "unknown")
        try:
            r = requests.post(f"{CONFIG['server_url']}/heartbeat", json={
                "worker_id": WORKER_ID, "status": "idle",
            }, timeout=10)
            r.raise_for_status()
            data = r.json()
            job = data.get("assigned_job")
            if job:
                span.set_attribute("job.assigned", job["job_id"])
            return job
        except Exception as e:
            span.set_attribute("error", str(e))
            _log(f"[WARN] Heartbeat failed: {e}")
            return None

def send_result(job_id, success, error=""):
    with tracer.start_as_current_span("worker.send_result") as span:
        span.set_attribute("job.id", job_id)
        span.set_attribute("success", success)
        try:
            requests.post(f"{CONFIG['server_url']}/result", json={
                "worker_id": WORKER_ID, "job_id": job_id, "success": success, "error": error,
            }, timeout=10)
        except Exception as e:
            _log(f"[FAIL] Could not send result: {e}")


# ─── Job Execution ───

def execute_job(job: dict):
    job_id = job["job_id"]
    task_type = job["task_type"]
    prompt = job.get("prompt", "")
    params = job.get("params", {})

    with tracer.start_as_current_span("worker.execute_job") as span:
        span.set_attribute("job.id", job_id)
        span.set_attribute("job.type", task_type)
        if prompt:
            span.set_attribute("job.prompt", prompt[:80])

        _log(f"\n{'='*50}")
        _log(f"[JOB] {job_id} | Type: {task_type}")
        if prompt:
            _log(f"      Prompt: {prompt[:80]}")
        _log(f"{'='*50}")

        start_ts = time.time()

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available on this machine!")

            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            span.set_attribute("worker.gpu", gpu_name)
            _log(f"[GPU] {gpu_name}")

            result = {}

            if task_type == "matrix-multiply" or task_type == "compute":
                size = params.get("size", 2048)
                span.set_attribute("matrix.size", size)
                _log(f"[WORK] Multiplying {size}x{size} matrices on GPU...")
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)

                c = a @ b
                torch.cuda.synchronize()

                start = time.time()
                c = a @ b
                torch.cuda.synchronize()
                elapsed = time.time() - start

                flops = 2 * size**3 / elapsed / 1e12
                span.set_attribute("compute.time_seconds", round(elapsed, 4))
                span.set_attribute("compute.tflops", round(flops, 2))

                result = {
                    "operation": f"{size}x{size} matrix multiply",
                    "time_seconds": round(elapsed, 4),
                    "tflops": round(flops, 2),
                    "device": gpu_name,
                }
                _log(f"[DONE] {result}")

            elif task_type == "tensor-info":
                result = {
                    "device": gpu_name,
                    "cuda_version": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                }
                _log(f"[DONE] GPU info: {result}")

            else:
                code = prompt or params.get("code", "")
                if not code:
                    raise ValueError(f"Unknown task '{task_type}'. Try: compute, tensor-info, or send code")
                _log(f"[WORK] Executing user code...")
                local_vars = {"torch": torch, "device": torch.device("cuda"),
                              "WORK_DIR": str(WORK_DIR), "job_id": job_id, "params": params}
                exec(code, local_vars)
                result = local_vars.get("result", "Code executed successfully")
                _log("[DONE] Code executed")
                span.add_event("custom_code_executed")

            # Save & report
            duration = time.time() - start_ts
            out_path = WORK_DIR / f"{job_id}_result.json"
            out_path.write_text(json.dumps(result, indent=2))
            send_result(job_id, success=True)
            jobs_executed.add(1)
            job_duration_histogram.record(duration)
            logger.info("job completed", extra={"job_id": job_id, "duration_s": round(duration, 2)})

        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            span.set_attribute("error", str(e)[:200])
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            _log(f"[ERROR] {error_msg}")
            send_result(job_id, success=False, error=error_msg)


# ─── Main Loop ───

def main():
    _log("=" * 50)
    _log("  GPU Pod Worker — OTel enabled")
    _log("  SigNoz @ localhost:4317")
    _log("=" * 50)

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
