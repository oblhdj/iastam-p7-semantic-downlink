"""demo/host_power.py: the demo process must not be left to Windows power throttling.

Why it exists: on 10 Oct 2026 the detector ran about 25 times slower (about 800 ms per call instead
of 30) while the app that had started it was minimised. Forcing the same state (EcoQoS) on a test
process reproduces it; the opt-out removes it. Speed itself is not asserted here -- the throttled
control run takes half a minute and depends on the machine -- only that the request is made, that
it overrides a throttled state, and that every demo entry point makes it.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demo"
sys.path.insert(0, str(DEMO))

import host_power  # noqa: E402

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling only")


def test_keep_full_speed_reports_in_words_and_is_idempotent():
    first = host_power.keep_full_speed()
    assert isinstance(first, str) and first.split(":")[0] in ("opted out", "not needed", "skipped", "unavailable")
    assert host_power.keep_full_speed() == first


@windows
def test_opt_out_overrides_a_throttled_process(monkeypatch):
    """The state Windows gives a background process on battery, forced on; then the opt-out."""
    monkeypatch.delenv("P7_DEMO_ALLOW_THROTTLING", raising=False)
    assert host_power._set(throttled=True) == 0 and host_power.state() == "throttled"
    assert host_power._opt_out().startswith("opted out") and host_power.state() == "opted out"


@windows
def test_the_switch_leaves_the_process_alone(monkeypatch):
    assert host_power._set(throttled=False) == 0
    monkeypatch.setenv("P7_DEMO_ALLOW_THROTTLING", "1")
    assert host_power._opt_out().startswith("skipped") and host_power.state() == "opted out"


def test_off_windows_it_is_a_no_op(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert host_power._opt_out().startswith("not needed") and host_power.state() == "unknown"


def test_every_demo_entry_point_opts_out_before_the_detector_runs():
    for rel in ("dashboard.py", "make_fallback_assets.py", "quickstart/run_demo.py"):
        src = (DEMO / rel).read_text(encoding="utf-8")
        assert "host_power.keep_full_speed()" in src, f"demo/{rel} does not opt out of power throttling"
    dash = (DEMO / "dashboard.py").read_text(encoding="utf-8")
    assert dash.index("host_power.keep_full_speed()") < dash.index("import demo_pipeline")
