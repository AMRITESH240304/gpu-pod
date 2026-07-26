# GPU Pod

Send GPU compute tasks from your Mac to your Windows laptop over your local network.

```
Mac (Client) ──submit job──▶ FastAPI Server ──assign──▶ Windows Laptop (GTX 1650)
     │                          (Mac)                       │
     ◀──poll result──────────────◀───────return result──────┘
```

## Quick Start

### 1. Start the server (on your Mac)

```bash
cd server
uv sync
uv run python main.py
```

### 2. Start the worker — two options (on Windows)

**Option A — Desktop GUI (recommended if you want live GPU stats):**
Copy the entire `client/` folder to your Windows laptop, then:

```powershell
cd client
uv venv --python 3.13
.venv\Scripts\activate
uv sync

set GPU_POD_SERVER_URL=http://<mac-ip>:8000

uv run python main.py
```

A GUI window opens. Click **"I'm a GPU Provider"** → Start Worker. You'll see live GPU utilization, VRAM, and temperature.

**Option B — Headless worker (lightweight, no GUI):**
Copy just the `worker/` folder to your Windows laptop, then:

```powershell
cd worker
uv venv --python 3.13
.venv\Scripts\activate
uv sync

set GPU_POD_SERVER_URL=http://<mac-ip>:8000
set GPU_POD_WORKER_NAME=my-gtx1650

python worker.py
```

You'll see:
```
[OK] Registered | ID: a1b2c3d4 | GPU: NVIDIA GeForce GTX 1650 | VRAM: 4.0GB
```

> ⚠️ Both `client/` and `worker/` have PyTorch pre-configured for CUDA 12.4 in their `pyproject.toml`. If your NVIDIA driver needs a different version (check with `nvidia-smi`), change `cu124` to `cu118`, `cu121`, etc.

### 3. Launch the desktop app (on your Mac)

```bash
cd client
uv sync
uv run python main.py
```

A GUI window will open. Enter the server IP, click Connect, then choose **"I'm a GPU User"** to submit compute jobs and see available GPUs.

---

## What's included

| Component | What it does |
|-----------|--------------|
| `server/` | FastAPI coordinator — queues jobs, tracks workers (SQLite) |
| `worker/` | Simple Python worker — registers, heartbeats, runs GPU tensor ops |
| `client/` | Desktop GUI — connect, choose provider or user, monitor/submit jobs |

## GUI screenshots

**Connect screen** — enter server IP and port
**Role select** — "GPU Provider" or "GPU User"
**Provider dashboard** — live GPU utilization bars, VRAM, temperature, worker log
**User dashboard** — available GPUs list, matrix size slider, job history with results

## Tips

- Both devices need to be on the same Wi-Fi
- Find your Mac's IP: `ipconfig getifaddr en0`
- Worker sends heartbeat every 5s; auto-assigns jobs on each heartbeat
- No port forwarding needed — worker always connects **outbound** to the server