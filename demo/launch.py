"""One command for each way of running the demo. Standard library only; works from any shell.

    python demo/launch.py                the full demo: the dashboard, live on CPU
    python demo/launch.py --fallback     the same page on the pre-generated results only
    python demo/launch.py --quickstart   the chain in the terminal, about 2 s, no browser
    python demo/launch.py --summary      the committed results on one screen (any Python)
    python demo/launch.py --gpu          the detector-in-the-loop run on the real Airbus split
    python demo/launch.py --check        report the environment and the fallback assets; start nothing

    --install    first install what the chosen path needs into THIS interpreter's environment
    --dry-run    print the command that would run, and stop
    --full       with --gpu: all 5,320 test tiles instead of a 400-tile sample
    --port N     with the dashboard: serve on port N instead of 8501

What each path needs is checked before anything starts, and a missing piece is named together with
the command that provides it. The dashboard also falls back to demo/fallback_assets/ by itself
when the live pipeline cannot run; --fallback forces that, which is how to rehearse it (the
environment-variable form, P7_DEMO_FORCE_FALLBACK=1, is written differently in every shell).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
sys.path.insert(0, str(DEMO))

REQUIREMENTS = {"dashboard": DEMO / "requirements-dashboard.txt",
                "quickstart": DEMO / "quickstart" / "requirements.txt"}
MODULES = {"dashboard": ("streamlit", "numpy", "cv2", "onnxruntime", "skyfield", "sgp4"),
           "fallback": ("streamlit",),                       # the static page needs nothing else
           "quickstart": ("numpy", "cv2", "onnxruntime", "skyfield", "sgp4"),
           "summary": (), "gpu": (), "check": ()}


def missing(mode: str) -> list[str]:
    return [m for m in MODULES[mode] if importlib.util.find_spec(m) is None]


def plan(mode: str, full: bool = False, port: int | None = None) -> tuple[list[str], dict]:
    """(command, extra environment) for one mode. Pure: nothing is started or checked here."""
    py = sys.executable
    if mode in ("dashboard", "fallback"):
        cmd = [py, "-m", "streamlit", "run", str(DEMO / "dashboard.py")]
        if port:
            cmd += ["--server.port", str(port)]
        return cmd, ({"P7_DEMO_FORCE_FALLBACK": "1"} if mode == "fallback" else {})
    if mode == "quickstart":
        return [py, str(DEMO / "quickstart" / "run_demo.py")], {}
    if mode == "summary":
        return [py, str(DEMO / "summary.py")], {}
    if mode == "gpu":
        return ["bash", str(DEMO / "run_demo.sh")] + (["--full"] if full else []), {}
    raise ValueError(f"unknown mode {mode!r}")


def check() -> int:
    """The environment report the dashboard shows, in the terminal, plus the fallback assets."""
    import demo_data as D
    import host_power
    env = D.environment()
    for r in env["rows"]:
        mark = "ok     " if r["ok"] else "absent " if r["optional"] else "MISSING"
        print(f"  {mark} {r['name']}: {r['detail']}")
        if not r["ok"]:
            print(f"          needed for {r['needed_for']}\n          fix: {r['fix']}")
    index = D.fallback_index()
    stale = [] if "error" in index else D.fallback_staleness(index)
    for line in stale:
        print(f"  STALE   fallback assets: {line}")
    print(f"  power throttling: {host_power.keep_full_speed()}")
    print("\nlive dashboard: " + ("ready" if env["live_ok"] else "NOT ready, the page will open on the static fallback")
          + "\nstatic fallback: " + ("ready" if "error" not in index and not stale else "NOT ready"))
    return 0 if env["live_ok"] and "error" not in index and not stale else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    for name in ("fallback", "quickstart", "summary", "gpu", "check"):
        g.add_argument(f"--{name}", dest="mode", action="store_const", const=name)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args(argv)
    mode = a.mode or "dashboard"
    if mode == "check":
        return check()

    req = REQUIREMENTS.get("quickstart" if mode == "quickstart" else "dashboard")
    if a.install and mode in ("dashboard", "fallback", "quickstart"):
        install = [sys.executable, "-m", "pip", "install", "-r", str(req)]
        print("installing:", " ".join(install))
        if not a.dry_run and subprocess.call(install, cwd=REPO):
            print("\nThe install failed (see above). Nothing was started.")
            return 2
    cmd, extra = plan(mode, a.full, a.port)
    if a.dry_run:
        print(" ".join(f"{k}={v}" for k, v in extra.items()), " ".join(cmd))
        return 0

    absent = missing(mode)
    if absent:
        rel = req.relative_to(REPO).as_posix()
        print(f"Cannot start the {mode} path: this Python ({sys.executable}) has no {', '.join(absent)}.\n"
              f"  install them:   python -m pip install -r {rel}\n"
              f"  or in one go:   python demo/launch.py {'--' + mode + ' ' if mode != 'dashboard' else ''}--install")
        if mode == "dashboard" and "streamlit" not in absent:
            print("  or show the pre-generated results now:   python demo/launch.py --fallback")
        elif "streamlit" in absent:
            print("  with no network at all:   python demo/launch.py --summary   (needs nothing)")
        return 2
    if mode == "gpu" and shutil.which("bash") is None:
        print("Cannot start the GPU path: `bash` is not on PATH (on Windows, run this from Git Bash).\n"
              "  The committed results on one screen:   python demo/launch.py --summary")
        return 2
    if mode == "gpu" and not (REPO / "code" / ".venv312").exists():
        print("Note: code/.venv312 (torch) is absent, so demo/run_demo.sh will print the committed "
              "results instead of running the detector. See docs/REPRODUCE.md, section 5.")
    try:
        return subprocess.call(cmd, cwd=REPO, env={**os.environ, **extra})
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError as e:
        print(f"Cannot start: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
