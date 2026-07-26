"""
GPU Pod Desktop GUI
====================
Three screens: Connect → Role Select → Provider Dashboard or User Dashboard.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime

import requests
import customtkinter as ctk
from tkinter import ttk, messagebox

from . import gpu_utils
from . import worker as worker_engine


class GPUApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("GPU Pod")
        self.root.geometry("900x650")
        self.root.minsize(800, 550)

        self.server_url = None
        self._poll_jobs_active = False
        self.worker_thread = None

        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

        self.show_connect()

    # ──────────────────── CONNECT SCREEN ────────────────────

    def show_connect(self):
        self._clear()
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(frame, text="GPU Pod", font=ctk.CTkFont(size=32, weight="bold")).pack(pady=(0, 5))
        ctk.CTkLabel(frame, text="Connect to your GPU coordinator server", font=ctk.CTkFont(size=14)).pack(pady=(0, 25))

        ip_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ip_frame.pack(pady=10)
        ctk.CTkLabel(ip_frame, text="Server IP:").pack(side="left", padx=(0, 10))
        self.ip_entry = ctk.CTkEntry(ip_frame, width=250, placeholder_text="e.g. 192.168.1.42")
        self.ip_entry.pack(side="left")
        self.ip_entry.insert(0, "127.0.0.1")

        port_frame = ctk.CTkFrame(frame, fg_color="transparent")
        port_frame.pack(pady=5)
        ctk.CTkLabel(port_frame, text="Port:").pack(side="left", padx=(0, 10))
        self.port_entry = ctk.CTkEntry(port_frame, width=100, placeholder_text="8000")
        self.port_entry.insert(0, "8000")

        self.connect_btn = ctk.CTkButton(frame, text="Connect", command=self._connect, height=40, width=200)
        self.connect_btn.pack(pady=25)

        self.status_label = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        self.root.bind("<Return>", lambda e: self._connect())

    def _connect(self):
        ip = self.ip_entry.get().strip()
        port = self.port_entry.get().strip()
        if not ip:
            self.status_label.configure(text="Enter a server IP", text_color="red")
            return
        url = f"http://{ip}:{port}" if port else f"http://{ip}:8000"

        self.connect_btn.configure(state="disabled", text="Connecting...")
        self.root.update()

        try:
            r = requests.get(f"{url}/", timeout=5)
            r.raise_for_status()
            self.server_url = url
            self.status_label.configure(text="Connected!", text_color="green")
            self.root.after(500, self.show_role_select)
        except Exception as e:
            self.status_label.configure(text=f"Connection failed: {e}", text_color="red")
            self.connect_btn.configure(state="normal", text="Connect")

    # ──────────────────── ROLE SELECT SCREEN ────────────────────

    def show_role_select(self):
        self._clear()
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        frame.place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(frame, text="Connected to Server", font=ctk.CTkFont(size=16)).pack(pady=(0, 5))
        ctk.CTkLabel(frame, text=self.server_url, font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(0, 30))

        ctk.CTkLabel(frame, text="What do you want to do?", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(0, 25))

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack()

        self.provider_btn = ctk.CTkButton(
            btn_frame, text="🔧 I'm a GPU Provider\n(Run worker on this machine)",
            command=self._try_show_provider, height=80, width=300, font=ctk.CTkFont(size=14),
            fg_color="#2e7d32", hover_color="#1b5e20",
        )
        self.provider_btn.pack(side="left", padx=15)

        self.user_btn = ctk.CTkButton(
            btn_frame, text="🚀 I'm a GPU User\n(Submit jobs from this machine)",
            command=self.show_user, height=80, width=300, font=ctk.CTkFont(size=14),
            fg_color="#1565c0", hover_color="#0d47a1",
        )
        self.user_btn.pack(side="left", padx=15)

        if not gpu_utils.check_torch_cuda():
            ctk.CTkLabel(frame, text="Note: PyTorch CUDA not found. Provider mode needs it.",
                         font=ctk.CTkFont(size=11), text_color="orange").pack(pady=(20, 0))

    def _try_show_provider(self):
        if not gpu_utils.check_torch_cuda():
            ret = messagebox.askyesno("PyTorch CUDA Required",
                "Provider mode requires PyTorch with CUDA.\n\n"
                "Install with:\n  uv add torch\n\n"
                "Install now on this machine?")
            if ret:
                from tkinter import messagebox as mb
                self._install_torch_and_show_provider()
            return
        self.show_provider()

    def _install_torch_and_show_provider(self):
        self._clear()
        frame = ctk.CTkFrame(self.container)
        frame.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(frame, text="Installing PyTorch with CUDA...\nPlease wait.", font=ctk.CTkFont(size=14)).pack(pady=20)
        self.root.update()

        os.system("uv pip install --index-url https://download.pytorch.org/whl/cu124 torch 2>&1")

        # Re-check
        import importlib
        importlib.reload(gpu_utils)
        if gpu_utils.check_torch_cuda():
            messagebox.showinfo("Success", "PyTorch installed with CUDA support!")
            self.show_provider()
        else:
            messagebox.showerror("Install failed", "Could not install PyTorch. Try manually:\nuv pip install --index-url https://download.pytorch.org/whl/cu124 torch")

    # ──────────────────── PROVIDER DASHBOARD ────────────────────

    def show_provider(self):
        self._clear()
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="GPU Provider Dashboard", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        self.provider_status = ctk.CTkLabel(header, text="● Idle", font=ctk.CTkFont(size=14), text_color="gray")
        self.provider_status.pack(side="right", padx=10)
        self.start_stop_btn = ctk.CTkButton(header, text="▶ Start Worker",
                                            command=self._toggle_worker, width=120)
        self.start_stop_btn.pack(side="right", padx=5)

        # Stats cards
        stats_frame = ctk.CTkFrame(frame)
        stats_frame.pack(fill="x", pady=(0, 10))

        self.gpu_name_label = ctk.CTkLabel(stats_frame, text="GPU: --", font=ctk.CTkFont(size=13))
        self.gpu_name_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(stats_frame, text="Utilization:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=10, sticky="w")
        self.util_bar = ctk.CTkProgressBar(stats_frame, width=300)
        self.util_bar.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        self.util_bar.set(0)
        self.util_label = ctk.CTkLabel(stats_frame, text="0%", font=ctk.CTkFont(size=12), width=40)
        self.util_label.grid(row=1, column=2, padx=5, sticky="w")

        ctk.CTkLabel(stats_frame, text="VRAM:", font=ctk.CTkFont(size=12)).grid(row=2, column=0, padx=10, sticky="w")
        self.vram_bar = ctk.CTkProgressBar(stats_frame, width=300, progress_color="#2196f3")
        self.vram_bar.grid(row=2, column=1, padx=5, pady=3, sticky="ew")
        self.vram_bar.set(0)
        self.vram_label = ctk.CTkLabel(stats_frame, text="0 / 0 MB", font=ctk.CTkFont(size=12), width=120)
        self.vram_label.grid(row=2, column=2, padx=5, sticky="w")

        ctk.CTkLabel(stats_frame, text="Temperature:", font=ctk.CTkFont(size=12)).grid(row=3, column=0, padx=10, sticky="w")
        self.temp_label = ctk.CTkLabel(stats_frame, text="-- °C", font=ctk.CTkFont(size=12))
        self.temp_label.grid(row=3, column=1, padx=5, sticky="w")

        self.wid_label = ctk.CTkLabel(stats_frame, text="Worker ID: --",
                                      font=ctk.CTkFont(size=12), text_color="gray")
        self.wid_label.grid(row=4, column=0, columnspan=3, padx=10, pady=(5, 0), sticky="w")
        stats_frame.columnconfigure(1, weight=1)

        # Log
        log_frame = ctk.CTkFrame(frame)
        log_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="Log", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.log_text = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(size=12), wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkButton(frame, text="← Back", command=self.show_role_select, width=80).pack(anchor="w", pady=(5, 0))

        self._log("Ready. Click 'Start Worker' to begin.")

    def _toggle_worker(self):
        if not worker_engine.WORKER_RUNNING:
            worker_engine.start(self.server_url, self._log, self._update_stats)
            self.start_stop_btn.configure(text="■ Stop Worker", fg_color="#c62828", hover_color="#b71c1c")
            self.provider_status.configure(text="● Running", text_color="#4caf50")
            self._log("[START] Worker started")
            self._poll_gpu_stats()
        else:
            worker_engine.stop()
            self.start_stop_btn.configure(text="▶ Start Worker",
                                          fg_color=("#3a7ebf", "#1f538d"),
                                          hover_color=("#325882", "#14375e"))
            self.provider_status.configure(text="● Stopped", text_color="gray")
            self._log("[STOP] Worker stopping...")

    def _poll_gpu_stats(self):
        if not worker_engine.WORKER_RUNNING:
            return
        stats = gpu_utils.get_gpu_stats_nvidia()
        if stats:
            self._update_stats(stats)
        self.root.after(2000, self._poll_gpu_stats)

    def _update_stats(self, stats):
        self.gpu_name_label.configure(text=f"GPU: {stats['name']}")
        self.util_bar.set(stats['util'] / 100.0)
        self.util_label.configure(text=f"{stats['util']:.0f}%")
        mem_pct = stats['mem_used'] / stats['mem_total'] if stats['mem_total'] > 0 else 0
        self.vram_bar.set(mem_pct)
        self.vram_label.configure(text=f"{stats['mem_used']:.0f} / {stats['mem_total']:.0f} MB")
        self.temp_label.configure(text=f"{stats['temp']:.0f} °C")
        wid = worker_engine.get_worker_id()
        if wid:
            self.wid_label.configure(text=f"Worker ID: {wid}")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    # ──────────────────── USER DASHBOARD ────────────────────

    def show_user(self):
        self._clear()
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=15, pady=10)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text="GPU User Dashboard", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="↻ Refresh", command=self._refresh_user, width=100).pack(side="right", padx=5)
        ctk.CTkButton(header, text="← Back", command=self.show_role_select, width=80).pack(side="right", padx=5)

        main = ctk.CTkFrame(frame, fg_color="transparent")
        main.pack(fill="both", expand=True, pady=10)

        # LEFT: GPUs + controls
        left = ctk.CTkFrame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(left, text="Available GPUs", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)
        self.gpu_listbox = ctk.CTkTextbox(left, font=ctk.CTkFont(size=12), height=100)
        self.gpu_listbox.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(left, text="Matrix Size:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=10, pady=(10, 0))
        size_frame = ctk.CTkFrame(left, fg_color="transparent")
        size_frame.pack(fill="x", padx=10, pady=5)
        self.size_slider = ctk.CTkSlider(size_frame, from_=512, to=16384, number_of_steps=31,
                                         command=self._size_changed)
        self.size_slider.set(4096)
        self.size_slider.pack(side="left", fill="x", expand=True)
        self.size_label = ctk.CTkLabel(size_frame, text="4096", font=ctk.CTkFont(size=13), width=60)
        self.size_label.pack(side="right", padx=(10, 0))

        self.submit_btn = ctk.CTkButton(left, text="🚀 Submit Compute Job",
                                        command=self._submit_job, height=40)
        self.submit_btn.pack(fill="x", padx=10, pady=15)
        self.user_status = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=12))
        self.user_status.pack(pady=5)

        # RIGHT: Job history
        right = ctk.CTkFrame(main)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="Jobs History", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        columns = ("id", "type", "status", "time")
        self.job_tree = ttk.Treeview(right, columns=columns, show="headings", height=10)
        self.job_tree.heading("id", text="Job ID")
        self.job_tree.heading("type", text="Type")
        self.job_tree.heading("status", text="Status")
        self.job_tree.heading("time", text="Time (s)")
        self.job_tree.column("id", width=80)
        self.job_tree.column("type", width=80)
        self.job_tree.column("status", width=70)
        self.job_tree.column("time", width=70)
        self.job_tree.pack(fill="both", expand=True, padx=10, pady=5)
        self.job_tree.bind("<Double-1>", self._show_job_detail)

        ctk.CTkLabel(right, text="Result:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.result_text = ctk.CTkTextbox(right, font=ctk.CTkFont(size=12), height=120)
        self.result_text.pack(fill="x", padx=10, pady=5)

        self._refresh_user()
        self._poll_jobs_active = True
        self._poll_user_jobs()

    def _size_changed(self, val):
        sizes = [512, 1024, 2048, 4096, 8192, 16384]
        closest = min(sizes, key=lambda x: abs(x - int(val)))
        self.size_slider.set(closest)
        self.size_label.configure(text=str(closest))

    def _refresh_user(self):
        try:
            r = requests.get(f"{self.server_url}/workers", timeout=5)
            r.raise_for_status()
            workers = r.json()["workers"]
            self.gpu_listbox.delete("0.0", "end")
            if not workers:
                self.gpu_listbox.insert("0.0", "No workers registered.\nStart a worker from another machine!")
            else:
                for w in workers:
                    sym = "🟢" if w['status'] == 'idle' else "🟡" if w['status'] == 'busy' else "🔴"
                    self.gpu_listbox.insert("end",
                        f"{sym} {w['name']} | {w['gpu_model']} | {w['vram_gb']}GB | {w['status']}\n")
        except Exception as e:
            self.gpu_listbox.delete("0.0", "end")
            self.gpu_listbox.insert("0.0", f"Error: {e}")

    def _submit_job(self):
        try:
            size = int(self.size_label.cget("text"))
        except ValueError:
            size = 4096

        self.submit_btn.configure(state="disabled", text="Submitting...")
        self.user_status.configure(text="")

        def submit():
            try:
                r = requests.post(f"{self.server_url}/submit-job", json={
                    "task_type": "compute", "prompt": "", "params": {"size": size},
                }, timeout=10)
                r.raise_for_status()
                job_id = r.json()["job_id"]
                self.user_status.configure(text=f"✅ Submitted: {job_id}", text_color="green")
                self._log_result(f"Job {job_id} submitted ({size}x{size})\nWaiting for result...\n")
                self._poll_single_job(job_id)
            except Exception as e:
                self.user_status.configure(text=f"❌ {e}", text_color="red")
            finally:
                self.submit_btn.configure(state="normal", text="🚀 Submit Compute Job")

        threading.Thread(target=submit, daemon=True).start()

    def _poll_single_job(self, job_id):
        def poll():
            while True:
                try:
                    r = requests.get(f"{self.server_url}/job/{job_id}", timeout=5)
                    r.raise_for_status()
                    job = r.json()["job"]
                    if job["status"] in ("completed", "failed"):
                        self._add_job_to_tree(job)
                        if job["status"] == "completed":
                            self._log_result(f"✅ {job_id} completed!\nCheck 'Jobs History' for details.")
                        else:
                            self._log_result(f"❌ {job_id} failed: {job.get('error', 'unknown')}")
                        return
                except Exception:
                    pass
                time.sleep(2)
        threading.Thread(target=poll, daemon=True).start()

    def _add_job_to_tree(self, job):
        elapsed = ""
        if job.get("started_at") and job.get("completed_at"):
            try:
                s = datetime.fromisoformat(job["started_at"])
                e = datetime.fromisoformat(job["completed_at"])
                elapsed = f"{(e - s).total_seconds():.1f}"
            except Exception:
                pass
        self.job_tree.insert("", 0, values=(job["id"][:8], job["task_type"], job["status"], elapsed))

    def _poll_user_jobs(self):
        if not self._poll_jobs_active:
            return
        try:
            r = requests.get(f"{self.server_url}/jobs", timeout=5)
            r.raise_for_status()
            for item in self.job_tree.get_children():
                self.job_tree.delete(item)
            for j in r.json()["jobs"][:20]:
                self._add_job_to_tree(j)
        except Exception:
            pass
        self.root.after(5000, self._poll_user_jobs)

    def _show_job_detail(self, event):
        sel = self.job_tree.selection()
        if not sel:
            return
        item = self.job_tree.item(sel[0])
        job_id = item["values"][0]
        try:
            r = requests.get(f"{self.server_url}/job/{job_id}", timeout=5)
            r.raise_for_status()
            self._log_result(json.dumps(r.json()["job"], indent=2))
        except Exception as e:
            self._log_result(f"Error: {e}")

    def _log_result(self, text):
        self.result_text.delete("0.0", "end")
        self.result_text.insert("0.0", text)

    # ──────────────────── HELPERS ────────────────────

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        worker_engine.stop()
        self._poll_jobs_active = False
        self.root.destroy()
