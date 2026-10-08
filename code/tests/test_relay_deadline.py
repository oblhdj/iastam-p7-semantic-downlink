from datetime import datetime, timedelta, timezone

from sat7.orbit import Pass
from sat7.relay import ISLWindow, route_item

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _pass(h):
    t = START + timedelta(hours=h)
    return Pass(t, t, t + timedelta(minutes=8), 45.0, 1e9)


def _isl(h):
    return ISLWindow(START + timedelta(hours=h), START + timedelta(hours=h, minutes=5), 500.0, 600.0)


# direct ground pass is far (10 h); the relay reaches the ground much sooner (ISL now -> relay pass 2 h)
DIRECT = [_pass(10)]
ISL = [_isl(0.1)]
RELAY = [_pass(2)]


def test_without_deadline_min_J_picks_cheapest():
    # lam_E only, placeholder energies: relay (ISL+GS) costs more than direct -> direct wins on J
    out = route_item(1e6, START, DIRECT, ISL, RELAY, lam_E=1.0, lam_T=0.0)
    assert out["path"] == "direct" and out["deadline_missed"] is False


def test_deadline_disqualifies_the_slow_direct_path():
    # 5 h deadline: direct (10 h) is infeasible, relay (2 h) makes it -> relay chosen despite higher J
    out = route_item(1e6, START, DIRECT, ISL, RELAY, lam_E=1.0, lam_T=0.0, deadline_s=5 * 3600)
    assert out["path"] == "relay" and out["deadline_missed"] is False


def test_impossible_deadline_takes_the_fastest_and_flags_it():
    # 1 h deadline: neither path makes it -> pick the lowest-latency (relay) and flag the miss
    out = route_item(1e6, START, DIRECT, ISL, RELAY, lam_E=1.0, lam_T=0.0, deadline_s=1 * 3600)
    assert out["path"] == "relay" and out["deadline_missed"] is True


def test_deadline_met_by_direct_keeps_min_J():
    # generous 12 h deadline: both feasible -> back to min(J), direct
    out = route_item(1e6, START, DIRECT, ISL, RELAY, lam_E=1.0, lam_T=0.0, deadline_s=12 * 3600)
    assert out["path"] == "direct" and out["deadline_missed"] is False
