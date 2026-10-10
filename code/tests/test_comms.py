"""sat7.comms: the two-path link simulator [SIM] -- capacity, transmit time, energy, routing."""
from datetime import datetime, timedelta, timezone

import pytest

from sat7.comms import LABEL, CommsConfig, Links, pass_rate_bps, simulate_comms
from sat7.orbit import Pass
from sat7.relay import ISLWindow
from sat7.scheduler import FIFO, Item, NoBuffer, ValueGreedy, simulate

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _pass(h, cap=1e6, dur=400):
    t = START + timedelta(hours=h)
    return Pass(t, t, t + timedelta(seconds=dur), 45.0, cap)


def _isl(h, minutes=10):
    t = START + timedelta(hours=h)
    return ISLWindow(t, t + timedelta(minutes=minutes), 2000.0, 2500.0)


def _items(n=40, size=20_000.0, kind="P2"):
    return [Item(i, 600.0 * i, size, 1.0 + (i % 3), kind, (i,), progressive=kind != "P1") for i in range(n)]


# primary sees the ground at 10 h and 22 h; the relay at 2 h and 14 h; ISL every hour
LINKS = Links([_pass(10), _pass(22)], [_isl(0.5 + k) for k in range(20)], [_pass(2), _pass(14)])


@pytest.mark.parametrize("policy", [ValueGreedy(), FIFO(), NoBuffer()])
def test_relay_disabled_reduces_to_the_scheduler(policy):
    items = _items(60, 60_000.0)                         # more than a pass carries: truncation + backlog
    ref = simulate(items, LINKS.primary_passes, START, policy)
    got = simulate_comms(items, LINKS, START, policy, CommsConfig(relay_enabled=False)).result
    assert [(s.item.id, s.fraction, s.delivered_s) for s in got.sent] == \
           [(s.item.id, s.fraction, s.delivered_s) for s in ref.sent]
    assert got.bytes_sent == ref.bytes_sent and got.dropped_storage == ref.dropped_storage


def test_transmit_time_and_energy_come_from_bytes_and_the_physical_rate():
    # one 100 kB item, a 400 s pass carrying 1 MB at a 0.25 share: physical rate 80 kbit/s
    links = Links([_pass(1, cap=1e6)], [], [], primary_share=0.25)
    cr = simulate_comms([Item(0, 0.0, 100_000.0, 1.0, "P2", (0,))], links, START, ValueGreedy(),
                        CommsConfig(relay_enabled=False, p_tx_W=10.0))
    d = cr.deliveries[0]
    r_phys = pass_rate_bps(links.primary_passes[0], 0.25)
    assert r_phys == pytest.approx(1e6 * 8 / 400 / 0.25)
    assert d.tx_s["ground"] == pytest.approx(100_000 * 8 / r_phys)          # bytes * 8 / rate
    assert d.energy_J["ground"] == pytest.approx(10.0 * d.tx_s["ground"])   # power * duration
    # WHEN it lands uses the payload's share of the pass, not the physical rate
    assert d.delivered_s == pytest.approx(3600 + 400 * 100_000 / 1e6)


def test_relay_path_pays_both_stages_and_arrives_sooner():
    cfg = CommsConfig(isl_rate_bps=2e6, isl_efficiency=0.5, p_isl_W=12.0, p_tx_W=15.0)
    cr = simulate_comms(_items(10), LINKS, START, ValueGreedy(), cfg)
    relayed = [d for d in cr.deliveries if d.path == "relay"]
    assert relayed, "an item created at t=0 should reach the 2 h relay pass before the 10 h direct pass"
    d = relayed[0]
    assert set(d.tx_s) == {"isl", "relay_ground"}                            # both stages accounted
    assert d.tx_s["isl"] == pytest.approx(d.item.size * 8 / (2e6 * 0.5))
    assert d.energy_J["isl"] == pytest.approx(12.0 * d.tx_s["isl"])
    assert d.energy_J["relay_ground"] == pytest.approx(15.0 * d.tx_s["relay_ground"])
    direct = simulate_comms(_items(10), LINKS, START, ValueGreedy(), CommsConfig(relay_enabled=False))
    first = {x.item.id: x.delivered_s for x in direct.deliveries}
    assert d.delivered_s < first[d.item.id]


def test_links_are_never_overfilled():
    # 12 MB, all captured in the first hour, against 1 MB passes: every link saturates
    items = [Item(i, 9.0 * i, 30_000.0, 1.0 + (i % 3), "P2", (i,), progressive=True) for i in range(400)]
    cr = simulate_comms(items, LINKS, START, ValueGreedy(), CommsConfig(isl_rate_bps=1e5))
    assert all(v["used"] <= 1 + 1e-9 for v in cr.link_use.values())
    assert cr.link_use["relay_ground"]["used"] > 0.9     # the relay's passes are real extra capacity
    direct = simulate_comms(items, LINKS, START, ValueGreedy(), CommsConfig(relay_enabled=False))
    assert cr.result.bytes_sent > direct.result.bytes_sent


def test_isl_capacity_limits_what_the_relay_takes():
    # 1 kbit/s effective ISL for 10 min = 75 kB per window: at most 3 x 20 kB items per window
    cfg = CommsConfig(isl_rate_bps=1e3, isl_efficiency=1.0)
    cr = simulate_comms(_items(40), LINKS, START, ValueGreedy(), cfg)
    per_window = cfg.isl_effective_bps * 600 / 8
    assert cr.link_use["isl"]["used"] <= 1 + 1e-9
    assert sum(d.item.size for d in cr.deliveries if d.path == "relay") <= per_window * len(LINKS.isl)


def test_relay_only_forwards_processed_products_never_raw_imagery():
    raw = [Item(i, 60.0 * i, 20_000.0, 1.0, "RAW", (i,)) for i in range(10)]
    cr = simulate_comms(raw, LINKS, START, ValueGreedy(), CommsConfig())
    assert cr.deliveries and all(d.path == "direct" for d in cr.deliveries)


def test_energy_weight_and_deadline_steer_the_choice():
    items = _items(10)
    fast = simulate_comms(items, LINKS, START, ValueGreedy(), CommsConfig(lam_E=0.0))
    frugal = simulate_comms(items, LINKS, START, ValueGreedy(), CommsConfig(lam_E=1e9))
    assert fast.totals()["relay_fraction_items"] > 0
    assert frugal.totals()["relay_fraction_items"] == 0          # relay always costs more joules
    # with a deadline, energy-frugal routing must still use the relay where direct would miss it
    dl = simulate_comms(items, LINKS, START, ValueGreedy(), CommsConfig(lam_E=1e9, deadline_s=4 * 3600))
    assert dl.totals()["relay_fraction_items"] > 0
    assert dl.totals()["deadline_met_fraction"] > simulate_comms(
        items, LINKS, START, ValueGreedy(), CommsConfig(relay_enabled=False, deadline_s=4 * 3600)
    ).totals()["deadline_met_fraction"]


def test_isl_setup_time_and_share_shrink_the_window():
    cfg = CommsConfig(isl_setup_s=300.0, isl_share=0.5)          # half of each 10 min window, half share
    cr = simulate_comms(_items(10), LINKS, START, ValueGreedy(), cfg)
    cap = cfg.isl_rate_bps * cfg.isl_efficiency * 0.5 * 300 / 8 * len(LINKS.isl) / 1e6
    assert cr.link_use["isl"]["capacity_MB"] == pytest.approx(cap, rel=1e-3)


def test_everything_is_labelled_sim():
    cr = simulate_comms(_items(5), LINKS, START, ValueGreedy(), CommsConfig())
    assert LABEL == "SIM" and cr.label == "SIM" and LINKS.label == "SIM"
    assert cr.totals()["label"] == "SIM" and all(line.startswith("[SIM]") for line in cr.summary())
