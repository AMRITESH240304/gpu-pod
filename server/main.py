"""
GPU Pod - FastAPI Coordinator Server
=====================================
Central server that manages GPU workers, accepts jobs from clients,
assigns work to available workers, and returns results.

Run:  uv run python main.py
"""

import uuid
import json
import os
import time
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

# ── OTel ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_pod_otel import init_otel, tracer, logger, meter

# OTel FastAPI instrumentation
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ──────────────────────── Setup ────────────────────────

DB_PATH = Path("gpu_pod.db")
UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("results")

UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="GPU Pod Coordinator")

# ── OTel initialisation (traces + logs + metrics → SigNoz) ──
init_otel("gpu-pod-server", env=os.environ.get("GPU_POD_ENV", "development"))

# Auto-instrument FastAPI endpoints
FastAPIInstrumentor.instrument_app(app)

# Instrument outgoing HTTP requests (e.g. worker callbacks)
RequestsInstrumentor().instrument()

# Custom metrics
job_counter = meter.create_counter(
    "jobs.total", unit="1", description="Total jobs submitted"
)
job_completed_counter = meter.create_counter(
    "jobs.completed", unit="1", description="Jobs completed successfully"
)
job_failed_counter = meter.create_counter(
    "jobs.failed", unit="1", description="Jobs failed"
)
active_workers_gauge = meter.create_up_down_counter(
    "workers.active", unit="1", description="Currently active workers"
)
queue_depth_gauge = meter.create_up_down_counter(
    "jobs.queue_depth", unit="1", description="Pending jobs in queue"
)

# ──────────────────────── Database ────────────────────────

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workers (
            id TEXT PRIMARY KEY,
            name TEXT,
            gpu_model TEXT,
            vram_gb REAL,
            status TEXT DEFAULT 'idle',
            last_heartbeat TEXT,
            ip TEXT,
            registered_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            task_type TEXT,
            prompt TEXT,
            params TEXT,
            status TEXT DEFAULT 'pending',
            worker_id TEXT,
            result_file TEXT,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ──────────────────────── Models ────────────────────────

class WorkerRegister(BaseModel):
    name: str = "worker"
    gpu_model: str = "unknown"
    vram_gb: float = 0.0

class Heartbeat(BaseModel):
    worker_id: str
    status: str = "idle"
    gpu_util_percent: float = 0.0
    vram_free_gb: float = 0.0

class JobSubmit(BaseModel):
    task_type: str           # e.g. "image-generation", "text-generation", "detect"
    prompt: str = ""
    params: dict = {}

class JobResult(BaseModel):
    worker_id: str
    job_id: str
    success: bool = True
    error: str = ""

# ──────────────────────── In-Memory State ────────────────────────

worker_heartbeats: dict[str, float] = {}  # worker_id -> last heartbeat timestamp
job_queue: list[str] = []                  # pending job IDs
job_queue_lock = threading.Lock()

# ──────────────────────── Helpers ────────────────────────

def _worker_heartbeat_ts(worker_id: str):
    """Worker seen alive."""
    worker_heartbeats[worker_id] = time.time()

def _get_idle_worker() -> Optional[dict]:
    """Find a worker that is idle and recently alive."""
    now = time.time()
    conn = get_db()
    workers = conn.execute(
        "SELECT * FROM workers WHERE status='idle' ORDER BY last_heartbeat DESC"
    ).fetchall()
    conn.close()
    for w in workers:
        wid = w["id"]
        if wid in worker_heartbeats and (now - worker_heartbeats[wid]) < 30:
            return dict(w)
    return None

def _assign_pending_jobs():
    """Pull from the job queue and assign to any idle worker."""
    with job_queue_lock:
        while job_queue:
            worker = _get_idle_worker()
            if not worker:
                break
            job_id = job_queue.pop(0)
            conn = get_db()
            conn.execute(
                "UPDATE jobs SET status='assigned', worker_id=?, started_at=datetime('now') WHERE id=?",
                (worker["id"], job_id),
            )
            conn.execute("UPDATE workers SET status='busy' WHERE id=?", (worker["id"],))
            conn.commit()
            conn.close()
            print(f"[ASSIGN] Job {job_id} → Worker {worker['name']} ({worker['id']})")

def _sweep_dead_workers():
    """Mark workers as dead if no heartbeat for 60 seconds."""
    now = time.time()
    dead_ids = [
        wid for wid, ts in worker_heartbeats.items()
        if now - ts > 60
    ]
    if dead_ids:
        conn = get_db()
        for wid in dead_ids:
            conn.execute("UPDATE workers SET status='dead' WHERE id=?", (wid,))
        conn.commit()
        conn.close()
        for wid in dead_ids:
            print(f"[DEAD] Worker {wid} marked dead (no heartbeat for 60s)")

# ──────────────────────── Background Sweeper ────────────────────────

def _background_loop():
    while True:
        time.sleep(10)
        with tracer.start_as_current_span("background-sweep") as span:
            _sweep_dead_workers()
            _assign_pending_jobs()
            with job_queue_lock:
                span.set_attribute("queue.depth", len(job_queue))

threading.Thread(target=_background_loop, daemon=True).start()

# ──────────────────────── API Routes ────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "GPU Pod Coordinator"}

# ── Worker Routes ──

@app.post("/register-worker")
def register_worker(reg: WorkerRegister, request: Request):
    with tracer.start_as_current_span("register-worker") as span:
        worker_id = str(uuid.uuid4())[:8]
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now(timezone.utc).isoformat()

        conn = get_db()
        conn.execute(
            "INSERT INTO workers (id, name, gpu_model, vram_gb, status, last_heartbeat, ip, registered_at) "
            "VALUES (?, ?, ?, ?, 'idle', ?, ?, ?)",
            (worker_id, reg.name, reg.gpu_model, reg.vram_gb, now, client_ip, now),
        )
        conn.commit()
        conn.close()

        _worker_heartbeat_ts(worker_id)
        active_workers_gauge.add(1)
        span.set_attribute("worker.id", worker_id)
        span.set_attribute("worker.name", reg.name)
        span.set_attribute("worker.gpu", reg.gpu_model)
        logger.info("worker registered", extra={"worker_id": worker_id, "gpu": reg.gpu_model})

        print(f"[REGISTER] New worker: {reg.name} | GPU: {reg.gpu_model} | VRAM: {reg.vram_gb}GB | IP: {client_ip}")

        return {
            "worker_id": worker_id,
            "message": f"Worker '{reg.name}' registered successfully",
        }

@app.post("/heartbeat")
def heartbeat(hb: Heartbeat):
    with tracer.start_as_current_span("heartbeat") as span:
        now = datetime.now(timezone.utc).isoformat()
        _worker_heartbeat_ts(hb.worker_id)
        span.set_attribute("worker.id", hb.worker_id)
        span.set_attribute("worker.status", hb.status)

        conn = get_db()
        conn.execute(
            "UPDATE workers SET last_heartbeat=?, status=? WHERE id=?",
            (now, hb.status, hb.worker_id),
        )
        conn.commit()

        # If worker is idle, try to assign a pending job immediately
        if hb.status == "idle":
            with job_queue_lock:
                if job_queue:
                    job_id = job_queue.pop(0)
                    conn.execute(
                        "UPDATE jobs SET status='assigned', worker_id=?, started_at=datetime('now') WHERE id=?",
                        (hb.worker_id, job_id),
                    )
                    conn.execute("UPDATE workers SET status='busy' WHERE id=?", (hb.worker_id,))
                    conn.commit()
                    span.add_event("job.assigned", {"job_id": job_id, "worker_id": hb.worker_id})
                    queue_depth_gauge.add(-1)
                    print(f"[ASSIGN] Job {job_id} → Worker {hb.worker_id} (via heartbeat)")

        # Check if there's work for this worker (may have just been assigned)
        assigned = None
        row = conn.execute(
            "SELECT id, task_type, prompt, params FROM jobs WHERE worker_id=? AND status='assigned' LIMIT 1",
            (hb.worker_id,),
        ).fetchone()
        conn.close()

        if row:
            params_raw = row["params"]
            try:
                params_parsed = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            except (json.JSONDecodeError, TypeError):
                params_parsed = {}
            assigned = {
                "job_id": row["id"],
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "params": params_parsed,
            }
            span.set_attribute("job.assigned", row["id"])

        return {
            "ok": True,
            "assigned_job": assigned,
        }

@app.get("/workers")
def list_workers():
    conn = get_db()
    rows = conn.execute("SELECT * FROM workers ORDER BY last_heartbeat DESC").fetchall()
    conn.close()
    return {"workers": [dict(r) for r in rows]}

# ── Job Routes ──

@app.post("/submit-job")
def submit_job(job: JobSubmit):
    with tracer.start_as_current_span("submit-job") as span:
        job_id = str(uuid.uuid4())[:8]
        params_json = json.dumps(job.params)

        span.set_attribute("job.id", job_id)
        span.set_attribute("job.type", job.task_type)
        span.set_attribute("job.prompt", job.prompt[:60])

        conn = get_db()
        conn.execute(
            "INSERT INTO jobs (id, task_type, prompt, params, status) VALUES (?, ?, ?, ?, 'pending')",
            (job_id, job.task_type, job.prompt, params_json),
        )
        conn.commit()
        conn.close()

        with job_queue_lock:
            job_queue.append(job_id)
            queue_depth_gauge.add(1)

        job_counter.add(1)
        logger.info("job submitted", extra={"job_id": job_id, "type": job.task_type})
        print(f"[NEW JOB] {job_id} | Type: {job.task_type} | Prompt: {job.prompt[:60]}")

        # Try to assign immediately
        _assign_pending_jobs()

        return {"job_id": job_id, "status": "pending", "message": "Job submitted"}

@app.get("/job/{job_id}")
def get_job(job_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(row)
    # Parse params from JSON string back to dict
    if isinstance(job.get("params"), str):
        try:
            job["params"] = json.loads(job["params"])
        except (json.JSONDecodeError, TypeError):
            job["params"] = {}
    return {"job": job}

@app.get("/jobs")
def list_jobs(status: Optional[str] = None):
    conn = get_db()
    if status:
        rows = conn.execute("SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    jobs = [dict(r) for r in rows]
    for j in jobs:
        if isinstance(j.get("params"), str):
            try:
                j["params"] = json.loads(j["params"])
            except (json.JSONDecodeError, TypeError):
                j["params"] = {}
    return {"jobs": jobs}

# ── Result Route (worker sends back) ──

@app.post("/result")
async def submit_result(result: JobResult, request: Request):
    """Worker sends back the result of a completed job."""
    with tracer.start_as_current_span("job-result") as span:
        span.set_attribute("job.id", result.job_id)
        span.set_attribute("worker.id", result.worker_id)
        span.set_attribute("job.success", result.success)

        conn = get_db()

        job = conn.execute("SELECT * FROM jobs WHERE id=?", (result.job_id,)).fetchone()
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail="Job not found")

        if result.success:
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=datetime('now'), error=NULL WHERE id=?",
                (result.job_id,),
            )
            job_completed_counter.add(1)
        else:
            conn.execute(
                "UPDATE jobs SET status='failed', completed_at=datetime('now'), error=? WHERE id=?",
                (result.error, result.job_id),
            )
            job_failed_counter.add(1)
            span.set_attribute("job.error", result.error[:200])

        # Free the worker
        conn.execute("UPDATE workers SET status='idle' WHERE id=?", (result.worker_id,))
        active_workers_gauge.add(-1)
        conn.commit()
        conn.close()

        logger.info("job completed" if result.success else "job failed",
                    extra={"job_id": result.job_id, "success": result.success})
        print(f"[RESULT] Job {result.job_id} → {'COMPLETED' if result.success else 'FAILED'}")

        # Check if we can assign more pending jobs
        _assign_pending_jobs()

        return {"ok": True}

# ──────────────────────── Entrypoint ────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  GPU Pod Coordinator Server")
    print("  http://0.0.0.0:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  OTel → SigNoz @ localhost:4317")
    print("=" * 50)
    logger.info("server starting", extra={"port": 8000})
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        from gpu_pod_otel import shutdown as otel_shutdown
        otel_shutdown()