"""Compute governor for parallel research agents (Windows, stdlib only).

    python scripts/ops/governor.py run --mb 1200 [--slots heavy] [--threads 2] [--name WS07] -- <command ...>
        waits for a free slot of the pool AND enough free RAM (requested + headroom), caps BLAS / Polars / Torch threads,
        runs the command inside a Windows Job Object with a hard memory limit (the whole process tree; the OS refuses
        allocations past it and the job is killed on exit), logs start / end / peak to data/run/governor.jsonl.
    python scripts/ops/governor.py lock <name> -- <command ...>
        runs the command while holding a named machine-wide mutex (e.g. `repo` for registry / diary / git writes).
    python scripts/ops/governor.py status
        free RAM, slots in use, running jobs.

Pools: heavy (default 4 slots: fits, data loads, simulations) and light (8 slots: tests, small scripts). The defaults keep
headroom for the live desk and the user's own applications; the user's processes are never touched.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import msvcrt
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "run"
SLOTS = {"heavy": 4, "light": 8}
HEADROOM_MB = 1500
LOG = RUN / "governor.jsonl"

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_mb() -> int:
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys // (1024 * 1024))


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wt.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD),
                ("SchedulingClass", wt.DWORD)]


class EXT_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x1F0FFF


def make_job(mb: int):
    job = k32.CreateJobObjectW(None, None)
    info = EXT_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PRIORITY_CLASS
    info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS   # the desk and the user's apps keep priority
    info.JobMemoryLimit = mb * 1024 * 1024
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    return job


def job_peak_mb(job) -> int:
    info = EXT_LIMIT()
    k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info), None)
    return int(info.PeakJobMemoryUsed // (1024 * 1024))


def _try_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return f
    except OSError:
        f.close()
        return None


def acquire_slot(pool: str, mb: int, name: str):
    waited = 0.0
    while True:
        if free_mb() - mb >= HEADROOM_MB:
            for i in range(SLOTS[pool]):
                f = _try_lock(RUN / "slots" / f"{pool}{i}.lock")
                if f is not None:
                    return f, i, waited
        time.sleep(5)
        waited += 5
        if waited % 300 == 0:
            print(f"[governor] {name}: waiting {int(waited)}s for a {pool} slot / {mb} MB (free {free_mb()} MB)", file=sys.stderr, flush=True)


def log(rec: dict) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    with mutex("governor-log"):
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class mutex:
    def __init__(self, name: str) -> None:
        self.path = RUN / "locks" / f"{name}.lock"

    def __enter__(self):
        while True:
            self.f = _try_lock(self.path)
            if self.f is not None:
                return self
            time.sleep(0.5)

    def __exit__(self, *a):
        try:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.f.close()


def run(a) -> int:
    f, i, waited = acquire_slot(a.slots, a.mb, a.name)
    env = dict(os.environ)
    t = str(a.threads)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TORCH_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        env[k] = t
    env["AI_TRADER_GOVERNOR_SLOT"] = f"{a.slots}{i}"
    job = make_job(a.mb)
    t0 = time.time()
    log({"ts": t0, "event": "start", "name": a.name, "pool": a.slots, "slot": i, "mb": a.mb, "waited_s": waited, "cmd": a.cmd[:12]})
    cmd = list(a.cmd)
    exe = cmd[0] if ("/" in cmd[0] or "\\" in cmd[0]) and Path(cmd[0]).exists() else shutil.which(cmd[0])
    if exe is None:
        log({"ts": time.time(), "event": "error", "name": a.name, "slot": i, "error": f"not found: {cmd[0]}"})
        f.close()
        print(f"[governor] executable not found: {cmd[0]}", file=sys.stderr)
        return 127
    cmd[0] = str(Path(exe).resolve())
    try:
        p = subprocess.Popen(cmd, env=env)
    except OSError as e:
        log({"ts": time.time(), "event": "error", "name": a.name, "slot": i, "error": repr(e)})
        f.close()
        raise
    h = k32.OpenProcess(PROCESS_ALL_ACCESS, False, p.pid)
    k32.AssignProcessToJobObject(job, h)
    rc = p.wait()
    peak = job_peak_mb(job)
    log({"ts": time.time(), "event": "end", "name": a.name, "slot": i, "rc": rc, "secs": round(time.time() - t0, 1), "peak_mb": peak, "cap_mb": a.mb})
    if peak >= a.mb * 0.98:
        print(f"[governor] {a.name}: hit the {a.mb} MB cap (peak {peak} MB); allocations past it fail. Use less data per step or ask for more.", file=sys.stderr)
    k32.CloseHandle(h)
    k32.CloseHandle(job)
    f.close()
    return rc


def status() -> None:
    print(f"free RAM {free_mb()} MB (headroom kept {HEADROOM_MB} MB)")
    for pool, n in SLOTS.items():
        busy = []
        for i in range(n):
            g = _try_lock(RUN / "slots" / f"{pool}{i}.lock")
            if g is None:
                busy.append(i)
            else:
                g.close()
        print(f"{pool}: {len(busy)}/{n} busy {busy}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mb", type=int, default=1000)
    r.add_argument("--slots", choices=list(SLOTS), default="heavy")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--name", default="agent")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    lk = sub.add_parser("lock")
    lk.add_argument("lockname")
    lk.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd_name == "status":
        status()
        return 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("missing command after --")
    a.cmd = cmd
    if a.cmd_name == "run":
        if a.mb > 4000 or a.threads > 4:
            ap.error("one job may use at most 4000 MB and 4 threads")
        return run(a)
    exe = shutil.which(cmd[0]) or cmd[0]
    cmd[0] = str(Path(exe).resolve()) if Path(exe).exists() else exe
    with mutex(a.lockname):
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
