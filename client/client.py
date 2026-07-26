"""
GPU Pod - Mac Client
=====================
Submit GPU compute jobs to the coordinator and retrieve results.

Usage:
    python client.py compute --size 4096
    python client.py info
    python client.py submit --type tensor-info
    python client.py status <job_id>
    python client.py list
    python client.py workers
"""

import os, sys, time, json, argparse
import requests

SERVER_URL = os.environ.get("GPU_POD_SERVER_URL", "http://192.168.0.107:8000")


def cmd_compute(args):
    """Submit a matrix multiply benchmark."""
    payload = {
        "task_type": "compute",
        "prompt": "",
        "params": {"size": args.size},
    }
    r = requests.post(f"{SERVER_URL}/submit-job", json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"✅ Submitted compute job ({args.size}x{args.size} matrix multiply)")
    print(f"   Job ID: {data['job_id']}")
    _poll_and_show(data["job_id"])


def cmd_info(args):
    """Get GPU info from the worker."""
    payload = {"task_type": "tensor-info", "prompt": "", "params": {}}
    r = requests.post(f"{SERVER_URL}/submit-job", json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"✅ Submitted GPU info request")
    print(f"   Job ID: {data['job_id']}")
    _poll_and_show(data["job_id"])


def cmd_status(args):
    r = requests.get(f"{SERVER_URL}/job/{args.job_id}", timeout=10)
    if r.status_code == 404:
        print(f"❌ Job not found")
        sys.exit(1)
    r.raise_for_status()
    _print_job(r.json()["job"])


def cmd_list(args):
    params = {}
    if args.status:
        params["status"] = args.status
    r = requests.get(f"{SERVER_URL}/jobs", params=params, timeout=10)
    r.raise_for_status()
    jobs = r.json()["jobs"]
    if not jobs:
        print("No jobs.")
        return
    print(f"{'ID':<10} {'Type':<18} {'Status':<10} {'Created':<18} Result")
    print("-" * 90)
    for j in jobs:
        prompt = (j.get("prompt") or "")[:35]
        print(f"{j['id']:<10} {j['task_type']:<18} {j['status']:<10} {(j.get('created_at') or '')[:16]:<18} {prompt}")


def cmd_workers(args):
    r = requests.get(f"{SERVER_URL}/workers", timeout=10)
    r.raise_for_status()
    ws = r.json()["workers"]
    if not ws:
        print("No workers registered.")
        return
    print(f"{'ID':<10} {'Name':<20} {'GPU':<30} {'VRAM':<8} {'Status':<8}")
    print("-" * 80)
    for w in ws:
        print(f"{w['id']:<10} {w['name']:<20} {w['gpu_model'][:28]:<30} {f'{w['vram_gb']}GB':<8} {w['status']:<8}")


def _poll_and_show(job_id, interval=2):
    while True:
        try:
            r = requests.get(f"{SERVER_URL}/job/{job_id}", timeout=10)
            r.raise_for_status()
            job = r.json()["job"]
            status = job["status"]
            if status in ("completed", "failed"):
                print()
                _print_job(job)
                if status == "completed":
                    print("\n✅ Job completed!")
                else:
                    print(f"\n❌ Failed: {job.get('error', 'unknown')}")
                return
            print(f"  [{status}] waiting...", end="\r")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped. Check later: python client.py status", job_id)
            return


def _print_job(job):
    print(f"  Job ID    : {job['id']}")
    print(f"  Type      : {job['task_type']}")
    print(f"  Status    : {job['status']}")
    print(f"  Worker    : {job.get('worker_id') or 'N/A'}")
    print(f"  Created   : {job.get('created_at') or 'N/A'}")
    print(f"  Completed : {job.get('completed_at') or 'N/A'}")
    if job.get("error"):
        print(f"  Error     : {job['error'][:200]}")


def main():
    parser = argparse.ArgumentParser(description="GPU Pod Client")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("compute", help="Run GPU matrix multiply benchmark")
    p.add_argument("--size", type=int, default=2048, help="Matrix size (default: 2048)")

    sub.add_parser("info", help="Get GPU info from worker")

    p = sub.add_parser("status", help="Check job status")
    p.add_argument("job_id", help="Job ID")

    p = sub.add_parser("list", help="List jobs")
    p.add_argument("--status", "-s", help="Filter by status")

    sub.add_parser("workers", help="List registered workers")

    args = parser.parse_args()

    if args.cmd == "compute":
        cmd_compute(args)
    elif args.cmd == "info":
        cmd_info(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "workers":
        cmd_workers(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
