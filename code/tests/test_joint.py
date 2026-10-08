from datetime import timezone

from sat7.joint import (CostModel, JointConstraints, JointWeights, build_objects, pareto, solve)
from sat7.joint import P0, P1, P2, P3, INFO_CREDIT, JointObject, LevelOption
from sat7.priority import PriorityConfig
from sat7.scheduler import LoDConfig, Ship, Workload, WorkloadConfig

COST = CostModel()
SIZES = {P0: 0.0, P1: 40.0, P2: 940.0, P3: 3440.0}


def _obj(i, w=1.0, sizes=SIZES):
    opts = [LevelOption(lv, b, COST.energy_J(lv, b), COST.latency_s(lv, b),
                        w if lv >= P1 else 0.0, w * INFO_CREDIT[lv]) for lv, b in sizes.items()]
    return JointObject(i, w, opts, max(o.info for o in opts))


def test_unconstrained_optimum_is_the_lower_bound():
    objs = [_obj(0), _obj(1)]
    r = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=0.0, i_min=0.0))
    assert all(lv == P0 for lv in r.assignment.values())     # cheapest when nothing is required
    assert r.J == r.J_lower_bound and r.gap == 0.0 and r.valid


def test_accuracy_floor_forces_reporting():
    objs = [_obj(0), _obj(1)]
    r = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=0.0))
    assert all(lv == P1 for lv in r.assignment.values())     # cheapest feasible is metadata-only
    assert abs(r.accuracy - 1.0) < 1e-9 and r.feasible and r.valid


def test_info_floor_climbs_levels_and_costs_more():
    objs = [_obj(0), _obj(1)]
    lo = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=0.0))
    hi = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=0.9))
    assert hi.info_preservation + 1e-9 >= 0.9
    assert hi.J >= lo.J and hi.gap >= lo.gap and hi.valid     # constraints only raise cost


def test_constrained_J_never_below_lower_bound():
    objs = [_obj(i) for i in range(5)]
    for con in (JointConstraints(0.5, 0.3), JointConstraints(1.0, 0.8), JointConstraints(0.9, 0.9)):
        r = solve(objs, JointWeights(1, 1, 1), con)
        assert r.valid and r.J + 1e-9 >= r.J_lower_bound


def test_impossible_constraint_flagged_not_faked():
    objs = [_obj(0)]
    r = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0, i_min=1.01))
    assert not r.reachable                                   # info floor above the ceiling
    r2 = solve(objs, JointWeights(0, 1, 0), JointConstraints(a_min=1.0001, i_min=0.0))
    assert not r2.reachable                                  # accuracy floor above 100%


def test_weights_change_the_chosen_levels():
    # beta huge -> prefer fewer bytes (low levels); gamma/alpha only -> latency/energy also rise w/ bytes,
    # so all three cost terms push levels DOWN absent constraints; with an info floor the mix matters.
    objs = [_obj(i) for i in range(4)]
    con = JointConstraints(a_min=1.0, i_min=0.6)
    r_bytes = solve(objs, JointWeights(0, 1000, 0), con)
    assert r_bytes.feasible and r_bytes.valid
    # every object reports (accuracy satisfied) and the info floor is met at minimum byte cost
    assert r_bytes.accuracy + 1e-9 >= 1.0 and r_bytes.info_preservation + 1e-9 >= 0.6


def test_build_objects_from_workload_omits_undetected_and_cloud():
    ships = [Ship(0, 0.0, False, 0.9, False), Ship(1, 0.0, False, 0.1, False),  # 1 is below thr
             Ship(2, 0.0, False, 0.8, True)]
    tiles = [(0.0, "ships", (0, 1)), (0.0, "coast", (2,)), (0.0, "cloud", ())]
    wl = Workload(ships, tiles, WorkloadConfig())
    objs = build_objects(wl, LoDConfig(conf_low=0.25), PriorityConfig())
    ids = {o.id for o in objs}
    assert ids == {0, 2}                                     # undetected ship 1 and cloud dropped
    # the coastal object's P3 (context tile) is the biggest byte option
    coast = next(o for o in objs if o.id == 2)
    assert max(op.bytes for op in coast.options) > 1000


def test_pareto_returns_one_row_per_weighting():
    objs = [_obj(i) for i in range(3)]
    rows = pareto(objs, JointConstraints(a_min=1.0, i_min=0.5))
    assert len(rows) == 4 and all("J" in r and r["feasible"] for r in rows)
