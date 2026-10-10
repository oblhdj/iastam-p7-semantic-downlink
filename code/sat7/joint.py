"""The paper's joint multi-objective program, solved as one program (paper section VI, eq 20).

    minimise   J = alpha * E_total + beta * D_tx + gamma * T_total
    subject to Accuracy             >= A_min
               InformationPreservation >= I_min

`PAPER_COVERAGE.md` section 4 records an honest substitution: the repo optimised value-per-byte
(a per-item proxy) and the relay used the J = lam_E*E + lam_T*T path choice, but the *global*
alpha/beta/gamma program with the two constraints was never solved as one program. This module is
that program.

Formulation (separable objective, coupling constraints -- a multiple-choice knapsack):
  * Decision: for each detected object j, a semantic level x_j in {P0, P1, P2, P3} (priority.py).
  * Each level costs bytes D_j(x), energy E_j(x) = E_proc + E_comm, and latency T_j(x), and earns
    an accuracy credit (delivered at all) and an information credit (how richly), both rising with
    the level. Raising a level improves the two constraints and worsens the three cost terms -- the
    genuine tension the paper's program expresses.
  * Totals are sums over objects; the two constraints are aggregate fractions:
        Accuracy               = sum_j w_j * [x_j >= P1] / sum_j w_j          (weighted report rate)
        InformationPreservation = sum_j info_j(x_j)      / sum_j info_j(P3)   (richness retained)

Solver (the repo's own discipline -- a practical solution bracketed by a bound it must beat):
  * `j_lower_bound` is the per-object unconstrained argmin of J. The constraints only ever force
    levels UP, so no feasible assignment can cost less: it is a true lower bound on the constrained
    optimum. (Exactly the role fractional_bound / joint_upper_bound play for the scheduler.)
  * `solve` takes that unconstrained optimum, then if a constraint is unmet repairs it greedily,
    upgrading the object with the smallest extra-J per unit of constraint deficit closed, until both
    floors are met or the instance is proven infeasible (unreachable even at all-P3). The reported
    gap (J_solved - J_lower) / J_lower says how much the constraints cost over the free optimum.

Labels (project rule): D_tx bytes are SIM (the WP4/SizeModel sizes); T_total is SIM (bytes / the
orbit-sim link rate + stage times); E_total is TARGET unless real P_k/T_k are injected through
`CostModel` (sat7.energy supplies a calibrated one). alpha/beta/gamma, A_min, I_min are ASSUMPTION
-- the knobs the paper sweeps. This module reports no measured number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .priority import P0, P1, P2, P3, PriorityConfig, classify
from .scheduler import LoDConfig, Workload, _l1, _l2, _w, coast_tile_size


# ----------------------------------------------------------------------------- weights / constraints
@dataclass
class JointWeights:
    """alpha/beta/gamma in the paper's objective. ASSUMPTION; swept. Units make the three terms
    commensurable: energy in J, data in MB, latency in hours (so a 'typical' item contributes
    order-1 to each term at the default weights) -- state the units with any J you report."""
    alpha: float = 1.0         # weight on E_total (per joule)
    beta: float = 1.0          # weight on D_tx    (per megabyte)
    gamma: float = 1.0         # weight on T_total (per hour)


@dataclass
class JointConstraints:
    """The two floors. ASSUMPTION; swept."""
    a_min: float = 0.90        # at least 90% of (weighted) objects must be reported at all
    i_min: float = 0.50        # at least 50% of recoverable information must be preserved


# ----------------------------------------------------------------------------- per-object cost model
@dataclass
class CostModel:
    """Turns a level's bytes into energy and latency. Energy defaults are TARGET placeholders.

    Replace `p_proc_W`, `t_proc_s` and `p_tx_W`/`r_tx_bps` with calibrated values (sat7.energy /
    report 17/19) to make E real; until then E is explicitly TARGET. D and T are SIM: D is the
    encoder's own byte sizes, T = t_proc + bytes / r_tx (per-item comm time; ignores queueing, a
    documented lower bound on the scheduler's delivered latency).
    """
    p_tx_W: float = 15.0           # ASSUMPTION (report 17): transmit power while keying
    r_tx_bps: float = 2.53e6       # SIM: effective downlink rate (orbit sim, report 17)
    p_proc_W: float = 28.0         # ASSUMPTION (report 17): onboard processor power
    # per-level onboard processing time (s): richer levels cost more encode (crop + compress).
    # ASSUMPTION, physically-motivated, swept. P0 still pays the detector; the ladder is the extra.
    t_proc_s: dict = field(default_factory=lambda: {P0: 0.0, P1: 0.0005, P2: 0.004, P3: 0.012})

    def energy_J(self, level: int, bytes_: float) -> float:
        e_proc = self.p_proc_W * self.t_proc_s.get(level, 0.0)
        e_comm = self.p_tx_W * bytes_ / self.r_tx_bps        # P_tx * D / R
        return e_proc + e_comm

    def latency_s(self, level: int, bytes_: float) -> float:
        return self.t_proc_s.get(level, 0.0) + bytes_ / self.r_tx_bps


@dataclass
class LevelOption:
    """One admissible choice for one object: its level and everything the program needs about it."""
    level: int
    bytes: float
    energy_J: float
    latency_s: float
    acc: float                 # accuracy credit (object weight if level>=P1, else 0)
    info: float                # information credit (rises with level)


@dataclass
class JointObject:
    id: int
    weight: float
    options: list[LevelOption]     # one per admissible level, P0 first
    info_max: float                # info credit at the richest available level (normaliser)


# ----------------------------------------------------------------------------- building objects
#   information credit per level: metadata < +ROI < +context, in [0, 1] before weighting. These are
#   ASSUMPTION (swept) and deliberately the same shape the LoD/P-scheme values use.
INFO_CREDIT = {P0: 0.0, P1: 0.5, P2: 0.8, P3: 1.0}


def build_objects(wl: Workload, lod: LoDConfig | None = None, pcfg: PriorityConfig | None = None,
                  cost: CostModel | None = None, levels=(P0, P1, P2, P3)) -> list[JointObject]:
    """Build the joint program's objects from a workload: one per detectable ship.

    Each object may be assigned any level in `levels`; its byte size at each level is the P-scheme
    size (metadata, +ROI, +context), its energy/latency come from `cost`, and its accuracy/info
    credits follow the level. Ships the detector never fired on (confidence below threshold) are
    omitted -- they are not decisions, they are already lost.
    """
    lod = lod or LoDConfig()
    pcfg = pcfg or PriorityConfig()
    cost = cost or CostModel()
    objs: list[JointObject] = []
    for idx, (t, ctx, ids) in enumerate(wl.tiles):
        if ctx == "cloud":
            continue
        for i in ids:
            s = wl.ships[i]
            if s.confidence < lod.conf_low:
                continue
            w = _w(s, lod)
            meta, roi = lod.l0_bytes, _l1(s, lod)
            context = coast_tile_size(wl, idx, lod) if ctx == "coast" else _l2(s, lod)
            size_at = {P0: 0.0, P1: meta, P2: meta + roi, P3: meta + roi + context}
            opts = []
            for lv in levels:
                b = size_at[lv]
                opts.append(LevelOption(
                    level=lv, bytes=b,
                    energy_J=cost.energy_J(lv, b), latency_s=cost.latency_s(lv, b),
                    acc=w if lv >= P1 else 0.0, info=w * INFO_CREDIT[lv]))
            info_max = max(o.info for o in opts)
            objs.append(JointObject(id=i, weight=w, options=opts, info_max=info_max))
    return objs


# ----------------------------------------------------------------------------- objective + solve
MB = 1e6
HOUR = 3600.0


def _obj_J(opt: LevelOption, jw: JointWeights) -> float:
    """Per-object J in the weights' stated units (E in J, D in MB, T in hours)."""
    return jw.alpha * opt.energy_J + jw.beta * opt.bytes / MB + jw.gamma * opt.latency_s / HOUR


@dataclass
class JointResult:
    assignment: dict           # object id -> chosen level
    J: float
    E_total_J: float
    D_tx_MB: float
    T_total_h: float
    accuracy: float
    info_preservation: float
    feasible: bool             # both constraints met
    reachable: bool            # constraints could be met at all (not asking the impossible)
    J_lower_bound: float       # unconstrained per-object optimum (true lower bound)
    gap: float                 # (J - J_lower) / J_lower: the price the constraints impose

    @property
    def valid(self) -> bool:
        """The solved J can never beat the unconstrained lower bound (bracket check, like optimum.py)."""
        tol = 1e-9 * max(1.0, abs(self.J_lower_bound))
        return self.J + tol >= self.J_lower_bound


def _totals(objs, assign, jw):
    E = sum(_opt(o, assign[o.id]).energy_J for o in objs)
    D = sum(_opt(o, assign[o.id]).bytes for o in objs) / MB
    T = sum(_opt(o, assign[o.id]).latency_s for o in objs) / HOUR
    J = sum(_obj_J(_opt(o, assign[o.id]), jw) for o in objs)
    W = sum(o.weight for o in objs)
    A = sum(_opt(o, assign[o.id]).acc for o in objs) / W if W else 1.0
    Imax = sum(o.info_max for o in objs)
    I = sum(_opt(o, assign[o.id]).info for o in objs) / Imax if Imax else 1.0
    return J, E, D, T, A, I


def _opt(o: JointObject, level: int) -> LevelOption:
    for op in o.options:
        if op.level == level:
            return op
    raise KeyError(level)


def solve(objs: list[JointObject], weights: JointWeights | None = None,
          constraints: JointConstraints | None = None) -> JointResult:
    """Solve the joint program: unconstrained per-object optimum, then greedy constraint repair."""
    jw = weights or JointWeights()
    con = constraints or JointConstraints()
    if not objs:
        return JointResult({}, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, True, True, 0.0, 0.0)

    # 1) unconstrained optimum: each object at its own cheapest level. True lower bound on J.
    assign = {o.id: min(o.options, key=lambda op: _obj_J(op, jw)).level for o in objs}
    J_lower = sum(_obj_J(_opt(o, assign[o.id]), jw) for o in objs)

    # reachability: the best the two constraints could ever be (every object at its richest level)
    W = sum(o.weight for o in objs)
    Imax = sum(o.info_max for o in objs)
    A_ceiling = sum(o.weight for o in objs if any(op.level >= P1 for op in o.options)) / W if W else 1.0
    reachable = (con.a_min <= A_ceiling + 1e-12) and (con.i_min <= 1.0 + 1e-12) and Imax > 0

    # 2) greedy repair toward both floors, in two heap-driven phases. The objective is separable and
    #    both constraints are monotone in the level, so the repair decouples cleanly and runs in
    #    O(n log n) instead of the naive O(n^2) rescan:
    #      phase A -- accuracy: an object reports (acc jumps 0 -> w) exactly when it first reaches
    #                 >= P1, a one-off static cost. Lift the cheapest such objects until a_min.
    #      phase B -- information: raise info with single-level upgrades, cheapest cost-per-info
    #                 first (a heap), re-pushing an object's next step after each take, until i_min.
    #    A free or beneficial upgrade (dJ <= 0) is always worth taking, so it is applied eagerly.
    import heapq

    by_id = {o.id: o for o in objs}
    cur_level = dict(assign)
    # running accuracy / info sums (absolute, not yet normalised)
    A_sum = sum(_opt(by_id[i], cur_level[i]).acc for i in cur_level)
    I_sum = sum(_opt(by_id[i], cur_level[i]).info for i in cur_level)

    def opts_sorted(o):                               # levels low -> high, by level value
        return sorted(o.options, key=lambda op: op.level)

    def apply(o, target_level):
        nonlocal A_sum, I_sum
        a0, i0 = _opt(o, cur_level[o.id]).acc, _opt(o, cur_level[o.id]).info
        a1, i1 = _opt(o, target_level).acc, _opt(o, target_level).info
        A_sum += a1 - a0
        I_sum += i1 - i0
        cur_level[o.id] = target_level

    # eager free upgrades (never hurt J, and only help the constraints)
    for o in objs:
        levels = opts_sorted(o)
        cur_j = _obj_J(_opt(o, cur_level[o.id]), jw)
        for op in levels:
            if op.level > cur_level[o.id] and _obj_J(op, jw) <= cur_j + 1e-15:
                apply(o, op.level)
                cur_j = _obj_J(op, jw)

    # phase A: accuracy. Cheapest jump to the first level >= P1, for objects not yet reporting.
    if W and con.a_min > 0:
        cand = []
        for o in objs:
            if _opt(o, cur_level[o.id]).acc > 0:
                continue                              # already reports
            reach = [op for op in opts_sorted(o) if op.acc > 0]
            if reach:
                tgt = min(reach, key=lambda op: _obj_J(op, jw))   # cheapest reporting level
                dJ = _obj_J(tgt, jw) - _obj_J(_opt(o, cur_level[o.id]), jw)
                cand.append((dJ, o.id, tgt.level))
        cand.sort()
        for dJ, oid, lv in cand:
            if A_sum / W + 1e-12 >= con.a_min:
                break
            if cur_level[oid] < lv:
                apply(by_id[oid], lv)

    # phase B: information. Single-level upgrades by cheapest cost-per-info, via a heap.
    if Imax and con.i_min > 0:
        heap = []                                     # (dJ/dInfo, dJ, obj_id, from_level, to_level)

        def push_next(o):
            levels = opts_sorted(o)
            cur = cur_level[o.id]
            nxt = next((op for op in levels if op.level > cur), None)
            if nxt is None:
                return
            cur_op = _opt(o, cur)
            dI = nxt.info - cur_op.info
            if dI <= 0:
                return                                # no info to gain this step
            dJ = _obj_J(nxt, jw) - _obj_J(cur_op, jw)
            heapq.heappush(heap, (dJ / dI, dJ, o.id, cur, nxt.level))

        for o in objs:
            push_next(o)
        while heap and I_sum / Imax + 1e-12 < con.i_min:
            _, _, oid, frm, to = heapq.heappop(heap)
            if cur_level[oid] != frm:                 # stale entry (object moved via a free upgrade)
                push_next(by_id[oid])
                continue
            apply(by_id[oid], to)
            push_next(by_id[oid])

    assign = cur_level
    J, E, D, T, A, I = _totals(objs, assign, jw)
    feasible = (A + 1e-9 >= con.a_min) and (I + 1e-9 >= con.i_min)
    return JointResult(assignment=assign, J=J, E_total_J=E, D_tx_MB=D, T_total_h=T,
                       accuracy=A, info_preservation=I, feasible=feasible, reachable=reachable,
                       J_lower_bound=J_lower, gap=(J - J_lower) / J_lower if J_lower > 0 else 0.0)


def pareto(objs: list[JointObject], constraints: JointConstraints | None = None,
           grid=((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))) -> list[dict]:
    """Solve the program at several (alpha, beta, gamma) weightings -> the trade-off curve the paper
    promised (eqs 32-35). Each row is one corner of the E/D/T trade; feed a finer grid to sweep it."""
    out = []
    for a, b, g in grid:
        r = solve(objs, JointWeights(a, b, g), constraints)
        out.append({"alpha": a, "beta": b, "gamma": g, "J": r.J, "E_total_J": r.E_total_J,
                    "D_tx_MB": r.D_tx_MB, "T_total_h": r.T_total_h, "accuracy": r.accuracy,
                    "info_preservation": r.info_preservation, "feasible": r.feasible})
    return out


if __name__ == "__main__":
    # Self-check: a hand-built instance with a known answer, no real data.
    #   two objects, four levels each; byte sizes rise with level; default cost model.
    cost = CostModel()

    def mk(i, w, sizes):
        opts = [LevelOption(lv, b, cost.energy_J(lv, b), cost.latency_s(lv, b),
                            w if lv >= P1 else 0.0, w * INFO_CREDIT[lv])
                for lv, b in sizes.items()]
        return JointObject(i, w, opts, max(o.info for o in opts))

    sizes = {P0: 0.0, P1: 40.0, P2: 940.0, P3: 3440.0}
    objs = [mk(0, 1.0, sizes), mk(1, 1.0, sizes)]

    # no constraints, beta-only: both objects go to P0 (cheapest) -> accuracy 0, and that IS optimal
    r0 = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=0.0, i_min=0.0))
    assert all(lv == P0 for lv in r0.assignment.values()) and r0.accuracy == 0.0
    print(f"unconstrained beta-only -> all P0, J={r0.J:.3g}, valid={r0.valid}")

    # require 100% accuracy: both must reach >= P1; still cheapest feasible -> exactly P1
    r1 = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=0.0))
    assert all(lv == P1 for lv in r1.assignment.values()) and abs(r1.accuracy - 1.0) < 1e-9
    assert r1.J >= r0.J and r1.valid                 # constraints cannot lower J below the free optimum
    print(f"a_min=1.0 -> all P1, J={r1.J:.3g} >= {r0.J:.3g} (gap {r1.gap:.2%}), feasible={r1.feasible}")

    # require high information preservation -> levels climb further
    r2 = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=0.9))
    assert r2.info_preservation + 1e-9 >= 0.9 and r2.J >= r1.J and r2.valid
    print(f"i_min=0.9 -> J={r2.J:.3g}, info={r2.info_preservation:.2f}, feasible={r2.feasible}")

    # impossible ask: info floor above the richest level's ceiling is unreachable, flagged not faked
    r3 = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=1.01))
    assert not r3.reachable
    print(f"i_min=1.01 -> reachable={r3.reachable} (correctly infeasible)")
    print("joint self-check OK")
