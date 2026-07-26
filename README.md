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

### 3. Run a compute job (on your Mac)

```bash
cd client
uv sync
export GPU_POD_SERVER_URL=http://<mac-ip>:8000

# Run a 2048x2048 matrix multiply on the remote GPU
python client.py compute --size 2048

# Or just get GPU info
python client.py info
```

Output:
```
✅ Submitted compute job (2048x2048 matrix multiply)
   Job ID: e5f6g7h8
  [assigned] waiting...
  [completed] waiting...

  Job ID    : e5f6g7h8
  Type      : compute
  Status    : completed
  Worker    : a1b2c3d4

✅ Job completed!
```

---

## What's included

| Component | What it does |
|-----------|--------------|
| `server/` | FastAPI coordinator — queues jobs, tracks workers (SQLite) |
| `worker/` | Simple Python worker — registers, heartbeats, runs GPU tensor ops |
| `client/` | CLI — submit compute jobs, check results, list workers |

**No ML models, no Docker, no Redis.** Just PyTorch tensor operations on your GPU.

## CLI commands

```bash
python client.py compute --size 4096      # Run matrix multiply benchmark
python client.py info                      # Get GPU name/CUDA info
python client.py status <job_id>           # Check job status
python client.py list                      # List all jobs
python client.py workers                   # List registered workers
```

## What the worker runs

The worker does **simple GPU compute** — no model downloads, no Hugging Face:

- **`compute`** — Matrix multiplication benchmark (reports TFLOPS)
- **`tensor-info`** — GPU name, CUDA version, VRAM usage
- **Custom code** — Send any PyTorch code in the `prompt` field

When you want to add real AI models later, just edit `execute_job()` in `worker/worker.py`.

## Tips

- Both devices need to be on the same Wi-Fi
- Find your Mac's IP: `ipconfig getifaddr en0`
- Worker sends heartbeat every 5s; auto-assigns jobs on each heartbeat
- No port forwarding needed — worker always connects **outbound** to the server