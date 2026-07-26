# GPU Pod

Send GPU compute tasks from your Mac to your Windows laptop over your local network.

```
Mac (Client) ──submit job──▶ FastAPI Server ──assign──▶ Windows Laptop (GTX 1650)
     │                          (Mac)                       │
     ◀──poll result──────────────◀───────return result──────┘
     │                                                      │
     └────────────▶  SigNoz (Docker)  ◀────────────────────┘
                      Traces · Logs · Metrics
```

---

## Quick Start

### 0. Start SigNoz (once)

```bash
# Requires Docker. Start SigNoz observability stack:
castingcast

# SigNoz UI: http://localhost:8080
# OTLP gRPC: localhost:4317
```

All three components automatically send traces, logs, and metrics to SigNoz.

### 1. Start the server (on your Mac)

```bash
cd server
uv sync
uv run python main.py
```

### 2. Start the worker — two options (on Windows)

**Option A — Desktop GUI (recommended):**

```powershell
cd client
uv venv --python 3.13
.venv\Scripts\activate
uv sync
set GPU_POD_SERVER_URL=http://<mac-ip>:8000
uv run python main.py
```

Click **"I'm a GPU Provider"** → Start Worker.

**Option B — Headless worker:**

```powershell
cd worker
uv venv --python 3.13
.venv\Scripts\activate
uv sync
set GPU_POD_SERVER_URL=http://<mac-ip>:8000
python worker.py
```

### 3. Launch the client (on your Mac)

```bash
cd client
uv sync
uv run python main.py
```

Enter the server IP, Connect, then **"I'm a GPU User"**.

---

## OpenTelemetry — SigNoz

| Component | Service name | Traces | Logs | Metrics |
|-----------|-------------|--------|------|---------|
| **Server** | `gpu-pod-server` | FastAPI endpoints, job lifecycle | All server logs | Job count, queue depth, active workers |
| **Worker** | `gpu-pod-worker` | Register, heartbeat, job execution | Worker logs | Jobs executed, duration histogram |
| **Client** | `gpu-pod-client` | Connect, submit, provider actions | GUI actions | Jobs submitted |

All data is sent via OTLP gRPC to `localhost:4317`. Every span captures:
- Service name + version
- Job IDs, worker IDs, matrix sizes
- GPU model and TFLOPS
- Success/failure with error details

**View in SigNoz:** http://localhost:8080 → Services → pick a service → Traces / Logs / Metrics

---

## What's included

| Path | What it does |
|------|--------------|
| `server/main.py` | FastAPI coordinator + OTel instrumentation |
| `worker/worker.py` | Standalone GPU worker + OTel instrumentation |
| `client/main.py` | Desktop GUI entry point + OTel |
| `client/client/app.py` | Connect/Provider/User dasboards |
| `client/client/worker.py` | Worker engine for GUI Provider mode |
| `client/client/gpu_utils.py` | GPU detection helpers |
| `gpu_pod_otel.py` | Shared OTel init for all components |
| `casting.yaml` | SigNoz deployment config (Docker Compose) |

## Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `GPU_POD_SERVER_URL` | `http://localhost:8000` | Worker, Client |
| `GPU_POD_WORKER_NAME` | `gpu-worker` | Worker |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | All (SigNoz) |
| `GPU_POD_ENV` | `development` | Server |

## Tips

- Both devices need to be on the same Wi-Fi
- Find your Mac's IP: `ipconfig getifaddr en0`
- Worker auto-assigns jobs on each heartbeat (every 5s)
- SigNoz retains traces, logs, and metrics in ClickHouse
- Check dead workers with: `python client.py workers`
