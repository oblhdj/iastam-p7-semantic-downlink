from datetime import datetime, timedelta, timezone

from sat7.orbit import Pass
from sat7.scheduler import (FIFO, Item, NoBuffer, ValueGreedy, WorkloadConfig, encode_lod, generate_workload,
                            metrics, simulate)

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _pass(at_h: float, cap: float) -> Pass:
    t = START + timedelta(hours=at_h)
    return Pass(t, t + timedelta(minutes=4), t + timedelta(minutes=8), 45.0, cap)


def test_greedy_prefers_valuable_bytes():
    items = [Item(1, 0, 900, 1.0, "L1"), Item(2, 10, 100, 5.0, "L2")]
    res = simulate(items, [_pass(1, 500)], START, ValueGreedy())
    assert [s.item.id for s in res.sent] == [2]


def test_fifo_head_of_line_blocking():
    items = [Item(1, 0, 900, 1.0, "L1"), Item(2, 10, 100, 5.0, "L2")]
    res = simulate(items, [_pass(1, 500)], START, FIFO())
    assert res.sent == []              # the big old item blocks the queue


def test_progressive_truncation_fills_the_pass():
    items = [Item(1, 0, 1000, 4.0, "tile", progressive=True)]
    res = simulate(items, [_pass(1, 250)], START, ValueGreedy())
    assert len(res.sent) == 1 and abs(res.sent[0].fraction - 0.25) < 1e-9


def test_no_buffer_loses_old_items_but_fifo_keeps_them():
    items = [Item(1, 0, 100, 1.0, "L1")]
    passes = [_pass(1, 10), _pass(2, 1000)]   # first pass too small
    assert simulate(items, passes, START, NoBuffer()).sent == []
    assert len(simulate(items, passes, START, FIFO()).sent) == 1


def test_end_to_end_metrics_are_sane():
    wl = generate_workload(WorkloadConfig(tiles_per_day=2000, seed=3))
    items = encode_lod(wl)
    passes = [_pass(h, 5e6) for h in (3, 9, 15, 21)]
    m = metrics(simulate(items, passes, START, ValueGreedy(), encoder="LoD"), wl)
    assert 0 < m["ship_recall"] <= 1 and 0 <= m["dark_recall"] <= 1
    assert m["MB_sent"] <= m["MB_capacity"] + 1e-9
