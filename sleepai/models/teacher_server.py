"""The teacher as an on-demand llama-server: started when the teacher needs the GPU, stopped when
the student does (sleepai/gpu.py). Generation through llama-server is much faster than
in-process transformers, and stopping it gives the student the whole card."""
from __future__ import annotations

import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .. import gpu

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "tools" / "llama.cpp-bin" / "llama-server.exe"


class TeacherServer:
    def __init__(self, gguf: str | Path, port: int = 8093, slots: int = 4, ctx_per_slot: int = 4096,
                 log: str | Path | None = None, binary: str | Path = BIN):
        self.gguf, self.port, self.slots, self.ctx = Path(gguf), port, slots, ctx_per_slot
        self.binary, self.log_path = Path(binary), Path(log) if log else ROOT / "models" / "teacher-server.log"
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self._lock = threading.Lock()          # parallel teacher calls must not start two servers
        gpu.register("teacher", self.stop)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self) -> None:
        """Take the GPU for the teacher (the student's models are released) and start the server."""
        if self.running() and gpu.owner() == "teacher":
            return
        with self._lock:
            self._start()

    def _start(self) -> None:
        gpu.acquire("teacher")
        if self.running():
            return
        log = open(self.log_path, "a", encoding="utf-8")
        self.proc = subprocess.Popen([str(self.binary), "-m", str(self.gguf), "--port", str(self.port),
                                      "-ngl", "99", "--jinja", "-np", str(self.slots),
                                      "-c", str(self.ctx * self.slots)], stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2)
                print(f"[gpu] teacher server up in {time.time() - t0:.0f} s", flush=True)
                return
            except Exception:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"teacher llama-server did not start (see {self.log_path})")

    def stop(self) -> None:
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
