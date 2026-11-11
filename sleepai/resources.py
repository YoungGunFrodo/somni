"""Resource governor for sleep: training adapts to the machine instead of taking it over.

The PC belongs to the user. Sleep training reads what is free when it starts and keeps checking:

  - the PC is short on RAM (the user opened something)  -> PAUSE until memory frees up
  - the training process itself grew past its limit      -> clean caches, then train in smaller
                                                            micro-batches (same math: gradients are
                                                            accumulated), and if that is not enough,
                                                            SAVE and stop; the next run resumes
  - progress is saved regularly, so a stop, crash or reboot costs minutes, not the night

Real finding behind this: one training run grew to 36 GB of system RAM and was killed; an
earlier one took the GPU driver down. Neither may cost the user their PC or the night's work.
"""
from __future__ import annotations

import gc
import time

GB = 2 ** 30


class ResumableStop(RuntimeError):
    """Training saved its progress and stopped to protect the PC; run again to continue."""


class ResourceGovernor:
    def __init__(self, ram_limit_gb: float | None = None, min_free_gb: float = 3.0,
                 pause_max_s: float = 4 * 3600, log=print):
        import psutil
        self.psutil, self.proc, self.log = psutil, psutil.Process(), log
        avail = psutil.virtual_memory().available / GB
        own = self.rss()
        # Default limit: what the process uses now plus half of what is free, between 6 and 16 GB.
        self.ram_limit = ram_limit_gb or max(6.0, min(16.0, own + 0.5 * avail))
        self.min_free, self.pause_max = min_free_gb, pause_max_s
        self.paused_s = 0.0
        self.log(f"[resources] RAM free {avail:.1f} GB; training limit {self.ram_limit:.1f} GB; "
                 f"pause below {self.min_free:.1f} GB free")

    def rss(self) -> float:
        return self.proc.memory_info().rss / GB

    def free(self) -> float:
        return self.psutil.virtual_memory().available / GB

    @staticmethod
    def micro_batch_for_vram(batch_size: int) -> int:
        """Start with a micro-batch that fits the free VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return batch_size
            free = torch.cuda.mem_get_info()[0] / GB
        except Exception:
            return batch_size
        if free >= 12:
            return batch_size
        if free >= 8:
            return max(1, batch_size // 2)
        return 1

    def wait_if_pc_busy(self) -> None:
        """Yield to the user: pause while the PC is short on memory."""
        if self.free() >= self.min_free:
            return
        t0 = time.time()
        self.log(f"[resources] PC low on memory ({self.free():.1f} GB free): pausing")
        while self.free() < self.min_free + 1.0:
            if time.time() - t0 > self.pause_max:
                raise ResumableStop("PC stayed low on memory; progress saved")
            time.sleep(30)
        self.paused_s += time.time() - t0
        self.log(f"[resources] resumed after {time.time() - t0:.0f} s")

    def own_usage_ok(self) -> bool:
        """True if the process is within its limit (after cleaning caches if needed)."""
        if self.rss() <= self.ram_limit:
            return True
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.rss() <= self.ram_limit


def windows_gpu_used_bytes() -> int | None:
    """Dedicated VRAM in use by ALL programs (Windows performance counters), or None if
    unavailable. Needed because torch.cuda.mem_get_info() on Windows/ROCm reports only this
    process (measured: a second process saw 0.1 GB used while training held 10 GB)."""
    import platform
    import subprocess
    if platform.system() != "Windows":
        return None
    cmd = ("(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | "
           "Measure-Object CookedValue -Maximum | ForEach-Object { [int64]$_.Maximum }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def cap_vram(max_fraction: float = 0.7, headroom_gb: float = 1.0, log=print) -> float:
    """Cap this process's GPU memory to what is FREE now (minus headroom), at most max_fraction of
    the card. Measured: with the teacher's llama-server on the same GPU, a cap of 70% of the whole
    card over-committed VRAM and Windows moved the server's buffers into system RAM (11.3 GB),
    which starved the PC. Returns the fraction set (0 when there is no GPU)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return 0.0
    used_by_all = windows_gpu_used_bytes()
    if used_by_all is not None:              # card-wide; torch on Windows/ROCm sees only itself
        # The counter includes THIS process (the model it just loaded): only the other
        # programs' share is unavailable to us.
        # The fraction caps this process's TOTAL use (its loaded model included), so what it may
        # use is the card minus everyone else.
        others = max(0, used_by_all - torch.cuda.memory_reserved())
        free = max(0, total - others)
    frac =max(0.05, min(max_fraction, (free / GB - headroom_gb) / (total / GB)))
    torch.cuda.set_per_process_memory_fraction(frac)
    log(f"[resources] VRAM free {free / GB:.1f} of {total / GB:.1f} GB; training may use "
        f"{frac * total / GB:.1f} GB")
    return frac


def windows_restart_pending() -> bool:
    """True when Windows Update has a restart queued. A long run started now may be cut off
    by the automatic restart (this cost a finished 3-hour SQuAD training once)."""
    import sys
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except ImportError:
        return False
    keys = [r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"]
    for k in keys:
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, k))
            return True
        except OSError:
            pass
    return False


def warn_if_restart_pending() -> None:
    if windows_restart_pending():
        print("[resources] WARNING: Windows has a restart pending (Windows Update). Restart first, or pause "
              "updates / set active hours, or this run may be cut off.", flush=True)
