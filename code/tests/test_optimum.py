import math

from sat7.optimum import (best_rate, best_value, fractional_bound, item_value, measure_gap,
                          plan_value)
from sat7.scheduler import FIFO, Item, ValueGreedy


def _whole(i, size, value):
    return Item(i, 0, size, value, "L1")


def _prog(i, size, value):
    return Item(i, 0, size, value, "tile", progressive=True)


# ------------------------------------------------------------------ known answers


def test_dp_beats_greedy_on_the_textbook_counterexample():
    """Greedy by density takes the 7/6 item and stops; the optimum takes both 5s."""
    items = [_whole(1, 6, 7.0), _whole(2, 5, 5.0), _whole(3, 5, 5.0)]
    r = measure_gap(items, 10, ValueGreedy(), max_buckets=1000)
    assert r.greedy_value == 7.0
    assert r.optimal_value == 10.0
    assert abs(r.gap - 0.3) < 1e-9
    assert r.valid


def test_dp_matches_brute_force_on_small_random_instances():
    import itertools
    import random
    rng = random.Random(7)
    for _ in range(25):
        items = [_whole(i, rng.randint(1, 9), round(rng.uniform(0.5, 9), 2)) for i in range(7)]
        cap = rng.randint(5, 25)
        brute = max((sum(v for _, v in c)
                     for k in range(len(items) + 1)
                     for c in itertools.combinations([(i.size, i.value) for i in items], k)
                     if sum(s for s, _ in c) <= cap), default=0.0)
        dp, _, _ = best_value(items, cap, max_buckets=10_000, bucket_bytes=1.0)
        assert abs(dp - brute) < 1e-6, (items, cap, dp, brute)


def test_dp_matches_brute_force_with_progressive_items_too():
    """The bound is loose on progressive items, so it cannot verify them. Enumerate the
    discretised choices directly instead: every item picks skip or one of the fractions."""
    import itertools
    import random
    rng = random.Random(3)
    steps, floor = 3, 0.1
    fracs = [0.0] + [floor + (1 - floor) * k / steps for k in range(1, steps + 1)]
    for _ in range(15):
        items = []
        for i in range(5):
            size, value = rng.randint(2, 12), round(rng.uniform(0.5, 6), 2)
            items.append(_prog(i, size, value) if rng.random() < 0.6 else _whole(i, size, value))
        cap = rng.randint(6, 30)
        bucket = 1.0
        exact = bucketed = 0.0
        for combo in itertools.product(range(len(fracs)), repeat=len(items)):
            if any(fi > 1 for it, fi in zip(items, combo) if not it.progressive):
                continue                       # a whole item has only skip / take
            used_exact = used_buckets = val = 0.0
            for it, fi in zip(items, combo):
                f = fracs[fi] if it.progressive else (1.0 if fi else 0.0)
                used_exact += it.size * f
                if f > 0:                      # the DP charges whole buckets, rounded up
                    used_buckets += math.ceil(it.size * f / bucket) * bucket
                val += item_value(it, f)
            if used_exact <= cap:
                exact = max(exact, val)
            if used_buckets <= int(cap // bucket) * bucket:
                bucketed = max(bucketed, val)
        dp, _, _ = best_value(items, cap, max_buckets=10_000, frac_steps=steps,
                              trunc_floor=floor, bucket_bytes=bucket)
        # solving the *same* discretised instance, the DP must find exactly the optimum
        assert abs(dp - bucketed) < 1e-6, (dp, bucketed)
        # and rounding weights up can only cost value, never invent it
        assert dp <= exact + 1e-6, (dp, exact)


def test_everything_fits_means_no_gap():
    items = [_whole(1, 10, 3.0), _whole(2, 10, 4.0)]
    r = measure_gap(items, 1000, ValueGreedy(), max_buckets=500)
    assert r.gap == 0.0 and r.valid
    assert abs(r.optimal_value - 7.0) < 1e-9


def test_zero_capacity_delivers_nothing():
    r = measure_gap([_whole(1, 10, 3.0)], 0.0, ValueGreedy())
    assert r.optimal_value == 0.0 and r.gap == 0.0


# ------------------------------------------------------------------ the bound


def test_the_bound_really_is_an_upper_bound():
    """Randomised: the relaxation must dominate the DP on every instance."""
    import random
    rng = random.Random(11)
    for _ in range(40):
        items = []
        for i in range(12):
            size, value = rng.randint(1, 40), round(rng.uniform(0.2, 8), 2)
            items.append(_prog(i, size, value) if rng.random() < 0.4 else _whole(i, size, value))
        cap = rng.randint(10, 120)
        r = measure_gap(items, cap, ValueGreedy(), max_buckets=5000, bucket_bytes=1.0)
        assert r.valid, (r.greedy_value, r.optimal_value, r.bound)


def test_progressive_items_are_credited_at_their_best_rate():
    """sqrt is concave, so the first bytes of a progressive item are the richest ones;
    a bound that credits the linear rate would not be a bound at all."""
    p = _prog(1, 100, 10.0)
    w = _whole(2, 100, 10.0)
    assert best_rate(p, 0.1) > best_rate(w)
    assert abs(best_rate(p, 0.1) - 10.0 / 100 / math.sqrt(0.1)) < 1e-12
    # sending the truncation floor really does beat the linear rate
    assert item_value(p, 0.1) > p.value * 0.1


def test_bound_is_tight_when_everything_fits():
    items = [_whole(1, 3, 2.0), _whole(2, 4, 5.0)]
    assert abs(fractional_bound(items, 100) - 7.0) < 1e-9


# ------------------------------------------------------------------ plumbing


def test_plan_value_uses_the_same_convention_as_metrics():
    p = _prog(1, 100, 4.0)
    assert abs(plan_value([(p, 0.25)]) - 4.0 * 0.5) < 1e-12     # sqrt(0.25) = 0.5
    w = _whole(2, 100, 4.0)
    assert abs(plan_value([(w, 1.0)]) - 4.0) < 1e-12


def test_fifo_is_never_better_than_value_greedy_on_the_same_queue():
    items = [_whole(1, 90, 1.0), _whole(2, 10, 9.0)]
    cap = 50
    g = plan_value(ValueGreedy().plan(items, cap, 0.0))
    f = plan_value(FIFO().plan(items, cap, 0.0))
    assert g >= f


def test_finer_truncation_steps_never_lower_the_measured_optimum():
    """More options can only help the DP, so the optimum must be monotone in frac_steps --
    which is why the reported gap is a lower bound at coarse settings."""
    items = [_prog(i, 40, 3.0) for i in range(6)] + [_whole(9, 25, 4.0)]
    vals = [best_value(items, 70, max_buckets=5000, frac_steps=k, bucket_bytes=1.0)[0]
            for k in (1, 2, 4, 8, 16)]
    assert vals == sorted(vals)


# ------------------------------------------------------------------ whole-day bounds


def _pass(h, cap):
    from datetime import datetime, timedelta, timezone
    from sat7.orbit import Pass
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    t = start + timedelta(hours=h)
    return Pass(t, t + timedelta(minutes=4), t + timedelta(minutes=8), 45.0, cap), start


def test_joint_bound_dominates_every_feasible_pass_assignment():
    """Brute force over all (item -> window) assignments, honouring release times."""
    import itertools
    import random
    from sat7.optimum import joint_upper_bound
    rng = random.Random(5)
    for _ in range(120):
        passes, start = zip(*[_pass(h, rng.randint(800, 2500)) for h in (4, 10, 16)])
        start = start[0]
        items = [Item(i, rng.choice([0, 5 * 3600, 11 * 3600]), rng.randint(200, 1200),
                      round(rng.uniform(0.5, 5), 2), "L1") for i in range(6)]
        rises = [(p.rise - start).total_seconds() for p in passes]
        best = 0.0
        for assign in itertools.product([-1, 0, 1, 2], repeat=len(items)):
            load, val, ok = [0.0] * 3, 0.0, True
            for it, a in zip(items, assign):
                if a < 0:
                    continue
                if rises[a] < it.created_s:
                    ok = False
                    break
                load[a] += it.size
                val += it.value
            if ok and all(load[j] <= passes[j].capacity_bytes for j in range(3)):
                best = max(best, val)
        assert joint_upper_bound(items, list(passes), start) >= best - 1e-6


def test_clairvoyant_is_feasible_and_never_beats_the_bound():
    import random
    from sat7.optimum import clairvoyant_value, joint_upper_bound
    rng = random.Random(9)
    for _ in range(60):
        passes, start = zip(*[_pass(h, rng.randint(500, 4000)) for h in (3, 9, 15, 21)])
        start = start[0]
        items = []
        for i in range(20):
            size, value = rng.randint(100, 900), round(rng.uniform(0.3, 6), 2)
            created = rng.choice([0, 4 * 3600, 10 * 3600, 16 * 3600])
            items.append(Item(i, created, size, value, "tile", progressive=True)
                         if rng.random() < 0.3 else Item(i, created, size, value, "L1"))
        c = clairvoyant_value(items, list(passes), start)
        assert c <= joint_upper_bound(items, list(passes), start) + 1e-6


def test_an_item_cannot_use_a_window_that_already_opened():
    """A tile captured after the last pass has risen is undeliverable, whatever it is worth."""
    from sat7.optimum import clairvoyant_value, joint_upper_bound
    p, start = _pass(2, 10_000)
    late = [Item(1, 10 * 3600, 100, 99.0, "L1")]
    assert joint_upper_bound(late, [p], start) == 0.0
    assert clairvoyant_value(late, [p], start) == 0.0


def test_saturation_caps_an_item_at_its_own_value():
    from sat7.optimum import best_rate, saturation_bytes
    for it in (_whole(1, 500, 4.0), _prog(2, 500, 4.0)):
        assert abs(saturation_bytes(it) * best_rate(it) - it.value) < 1e-9
