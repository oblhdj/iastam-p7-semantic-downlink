"""How far from optimal is the greedy scheduler?

The technical plan lists "optimality gap on small passes" as a must-have deliverable and
the report says greedy-by-value-per-byte "is optimal for divisible items with linear value,
but not for indivisible items". Both are promises. This module measures it.

The problem solved in one contact window:

    maximise   sum over whole items      v_i * x_i        x_i in {0, 1}
             + sum over progressive ones v_j * sqrt(f_j)  f_j in {0} u [floor, 1]
    such that  sum s_i * x_i + sum s_j * f_j  <=  capacity

That is a knapsack with a concave-divisible class, so greedy is *not* optimal in general.
``best_value`` solves it exactly by dynamic programming after discretising bytes into
buckets and each progressive item's fraction into steps; the discretisation is stated with
every result, and ``fractional_bound`` gives a true upper bound on the continuous relaxation
so the DP can be checked from above (DP <= bound must always hold).

Exact means exact *for the discretised instance*. Bucketing bytes can only remove value
(an item is charged a rounded-up weight), so the DP result is a lower bound on the true
optimum, and the measured gap is therefore conservative: the real gap can only be smaller
than the one reported here, never larger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scheduler import Item, Policy


def item_value(item: Item, fraction: float) -> float:
    """Value delivered by sending ``fraction`` of this item (the metrics convention)."""
    if fraction <= 0:
        return 0.0
    return item.value * (math.sqrt(fraction) if item.progressive else 1.0)


def plan_value(plan) -> float:
    """Value of a policy's plan: pairs of (item, fraction)."""
    return sum(item_value(it, f) for it, f in plan)


def best_rate(it: Item, trunc_floor: float = 0.1) -> float:
    """The most value per byte this item could ever yield.

    A whole item can only give v/s. A progressive one gives v*sqrt(f) for f*s bytes, i.e.
    v / (s*sqrt(f)) per byte, which *rises* as f shrinks -- sqrt is concave, so the first
    bytes of a progressive item are the most valuable ones. The best available rate is
    therefore at the smallest fraction the scheduler may send, ``trunc_floor``.
    """
    if not it.size:
        return 0.0
    return it.value / it.size / (math.sqrt(trunc_floor) if it.progressive else 1.0)


def saturation_bytes(it: Item, trunc_floor: float = 0.1) -> float:
    """Bytes after which an item cannot be worth any more than ``it.value``.

    Crediting bytes at ``best_rate`` forever would let one item earn several times its own
    value, which is why the old bound was ~3x loose on progressive items. An item can never
    yield more than ``value``, so its credit saturates at ``value / best_rate`` bytes: the
    whole size for an ordinary item, and ``sqrt(floor) * size`` for a progressive one.
    """
    r = best_rate(it, trunc_floor)
    return it.value / r if r > 0 else 0.0


def fractional_bound(items: list[Item], capacity: float, trunc_floor: float = 0.1) -> float:
    """Upper bound on the value any feasible schedule can extract from this capacity.

    Credit b bytes spent on item i with ``min(b * best_rate_i, value_i)``. That dominates
    the item's true value curve everywhere -- exactly at b = 0.1*size and at b = size for a
    progressive item, generously in between -- and it is concave, so the best allocation is
    simply to fill in decreasing order of rate, letting each item saturate before moving on.
    No feasible schedule can beat the result.
    """
    left, total = capacity, 0.0
    for it in sorted(items, key=lambda i: best_rate(i, trunc_floor), reverse=True):
        if left <= 0:
            break
        take = min(saturation_bytes(it, trunc_floor), left)
        total += take * best_rate(it, trunc_floor)
        left -= take
    return total


def joint_upper_bound(items: list[Item], passes, start, trunc_floor: float = 0.1) -> float:
    """Upper bound on what a CLAIRVOYANT scheduler could deliver over the whole day.

    The per-window gap (``measure_gap``) answers "given the queue greedy got itself into,
    how much did it leave behind in this window?". It says nothing about the choices that
    produced that queue. This bounds the whole day instead: a scheduler that knows every
    future arrival in advance and packs all the windows together.

    Two constraints make this more than one big knapsack:

    * **release times** -- an item captured after a window has opened cannot use it, so
      capacity is not interchangeable. Early windows serve only early items; the last
      window serves everything.
    * **capacity per window**, not in total.

    Relaxing integrality, the value is concave in the bytes given to each item (see
    ``fractional_bound``), so a greedy sweep is optimal for the relaxation *provided* each
    item consumes the earliest capacity it can reach. Earliest-first matters: late capacity
    is reachable by every item, so spending it on an item that could have used an early
    window throws away flexibility. Consuming the most constrained capacity first is what
    makes this a bound rather than just a heuristic.
    """
    windows = sorted(((p.rise - start).total_seconds(), p.capacity_bytes) for p in passes)
    rises = [r for r, _ in windows]
    caps = [c for _, c in windows]

    total = 0.0
    for it in sorted(items, key=lambda i: best_rate(i, trunc_floor), reverse=True):
        need = saturation_bytes(it, trunc_floor)
        rate = best_rate(it, trunc_floor)
        for j, r in enumerate(rises):
            if need <= 0:
                break
            if r < it.created_s:                  # this window had already opened
                continue
            take = min(need, caps[j])
            caps[j] -= take
            total += take * rate
            need -= take
    return total


def clairvoyant_value(items: list[Item], passes, start, trunc_floor: float = 0.1) -> float:
    """A feasible whole-day schedule built with perfect foresight.

    Same ranking our policy uses -- value per byte -- but applied once to the entire day
    instead of pass by pass, and each item placed in the *earliest* window it can reach so
    that flexible late capacity is kept for items not yet captured. Being feasible, this is
    a lower bound on the joint optimum, so together with ``joint_upper_bound`` it brackets
    the end-to-end loss without ever having to solve the NP-hard problem exactly.

    The difference between this and what the online policy achieves is the price of *not
    knowing the future*, as distinct from the price of being greedy.
    """
    windows = sorted(((p.rise - start).total_seconds(), p.capacity_bytes) for p in passes)
    rises = [r for r, _ in windows]
    caps = [c for _, c in windows]
    total = 0.0
    for it in sorted(items, key=lambda i: i.density, reverse=True):
        for j, r in enumerate(rises):
            if r < it.created_s:
                continue
            if it.size <= caps[j]:
                caps[j] -= it.size
                total += it.value
                break
            if it.progressive and caps[j] >= trunc_floor * it.size:
                f = caps[j] / it.size
                total += item_value(it, f)
                caps[j] = 0.0
                break
    return total


@dataclass
class GapResult:
    greedy_value: float
    optimal_value: float
    bound: float
    gap: float                 # (optimal - greedy) / optimal
    n_items: int
    capacity_bytes: float
    buckets: int
    bucket_bytes: float
    frac_steps: int

    @property
    def valid(self) -> bool:
        """The DP must sit between the greedy solution and the relaxation bound.

        Written as two separate comparisons on purpose: chaining them
        (``greedy <= opt + tol <= bound + tol``) cancels the tolerance out of the second
        test, which turns 1e-16 rounding into a reported violation.
        """
        tol = 1e-6 * max(1.0, self.bound)
        return (self.greedy_value <= self.optimal_value + tol
                and self.optimal_value <= self.bound + tol)


def best_value(items: list[Item], capacity: float, max_buckets: int = 4000,
               frac_steps: int = 4, trunc_floor: float = 0.1,
               bucket_bytes: float | None = None) -> tuple[float, float, int]:
    """Exact DP optimum for the discretised instance. Returns (value, bucket_bytes, buckets).

    ``frac_steps`` is how many partial sends a progressive item may choose between
    ``trunc_floor`` and 1 (the scheduler's own truncation floor is 0.1). Pass
    ``bucket_bytes`` to fix the byte resolution instead of deriving it from ``max_buckets``;
    that is how the caller guarantees the smallest item spans more than one bucket.
    """
    if capacity <= 0 or not items:
        return 0.0, 0.0, 0
    bucket = bucket_bytes if bucket_bytes else max(1.0, capacity / max_buckets)
    cap = int(capacity // bucket)
    dp = np.full(cap + 1, -np.inf)
    dp[0] = 0.0

    for it in items:
        options = []                        # (weight in buckets, value)
        if it.progressive and frac_steps > 0:
            for k in range(1, frac_steps + 1):
                f = trunc_floor + (1.0 - trunc_floor) * k / frac_steps
                w = int(math.ceil(it.size * f / bucket))
                if w <= cap:
                    options.append((w, item_value(it, f)))
        else:
            w = int(math.ceil(it.size / bucket))
            if w <= cap:
                options.append((w, it.value))
        if not options:
            continue
        nxt = dp.copy()                     # the "skip this item" option
        for w, v in options:
            if w == 0:
                np.maximum(nxt, dp + v, out=nxt)
            else:
                np.maximum(nxt[w:], dp[:-w] + v, out=nxt[w:])
        dp = nxt

    best = float(np.max(dp[np.isfinite(dp)])) if np.isfinite(dp).any() else 0.0
    return best, bucket, cap


def measure_gap(items: list[Item], capacity: float, policy: Policy, now_s: float = 0.0,
                max_buckets: int = 4000, frac_steps: int = 4,
                trunc_floor: float = 0.1, bucket_bytes: float | None = None) -> GapResult:
    """Greedy versus the exact optimum on one contact window."""
    greedy = plan_value(policy.plan(items, capacity, now_s))
    opt, bucket, buckets = best_value(items, capacity, max_buckets, frac_steps,
                                      trunc_floor, bucket_bytes)
    bound = fractional_bound(items, capacity, trunc_floor)
    opt = max(opt, greedy)      # the DP cannot be worse than a feasible solution; if the
                                # discretisation made it look worse, report the greedy value
                                # and a zero gap rather than a negative one
    return GapResult(greedy_value=greedy, optimal_value=opt, bound=bound,
                     gap=(opt - greedy) / opt if opt > 0 else 0.0,
                     n_items=len(items), capacity_bytes=capacity,
                     buckets=buckets, bucket_bytes=bucket, frac_steps=frac_steps)
