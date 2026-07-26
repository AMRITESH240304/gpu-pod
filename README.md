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

### 2. Start the worker (on Windows)

Copy the `worker/` folder to your Windows laptop, then:

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

> **Note:** The `pyproject.toml` is pre-configured to pull the CUDA 12.4 build of PyTorch from `download.pytorch.org/whl/cu124`. If your NVIDIA driver supports a different CUDA version, change `cu124` in `pyproject.toml` to your version (e.g. `cu118`, `cu121`). Check with `nvidia-smi`.

### 3. Launch the desktop app (on your Mac)

```bash
cd client
uv sync
uv run python main.py
```

A GUI window will open. Enter the server IP, click Connect, then choose:

| Button | What it does |
|--------|--------------|
| **GPU Provider** | Runs the worker — shows live GPU stats (utilization, VRAM, temp) |
| **GPU User** | Submit compute jobs, see available GPUs, view results |

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