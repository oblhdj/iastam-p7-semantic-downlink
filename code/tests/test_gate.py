"""WP3 learned gate: the only trained artefact that had no test.

Two groups. The artefact tests read `results/wp3_gate_tradeoff.csv` and
`results/wp3_gate_val_selected.json` and pin the *claims* the report makes, so a re-train that
quietly makes the gate worse fails here. The model tests need torch and are skipped in `.venv`
(3.14, no torch); run them with `.venv312/Scripts/python.exe -m pytest tests/test_gate.py`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRADEOFF = RESULTS / "wp3_gate_tradeoff.csv"
VAL_SEL = RESULTS / "wp3_gate_val_selected.json"

# The classic pre-filter, measured on the same test tiles (WP3). The gate has to beat these.
CLASSIC = [(0.177, 0.645), (0.419, 0.555), (0.581, 0.333)]   # (empty_dropped, ship_recall)


@pytest.fixture(scope="module")
def tradeoff() -> pd.DataFrame:
    if not TRADEOFF.exists():
        pytest.skip(f"{TRADEOFF.name} missing -- run scripts/wp3_train_gate.py")
    return pd.read_csv(TRADEOFF).sort_values("threshold").reset_index(drop=True)


@pytest.fixture(scope="module")
def val_selected() -> dict:
    if not VAL_SEL.exists():
        pytest.skip(f"{VAL_SEL.name} missing")
    return json.loads(VAL_SEL.read_text())


# ----------------------------------------------------------------- shape of the trade-off


def test_tradeoff_is_monotone(tradeoff):
    """Raising the threshold can only drop more tiles and keep fewer ships."""
    assert tradeoff.ship_recall.is_monotonic_decreasing
    assert tradeoff.empty_dropped.is_monotonic_increasing
    assert tradeoff.tiles_kept.is_monotonic_decreasing


def test_tradeoff_endpoints(tradeoff):
    """Threshold 0 keeps everything; the curve stays a valid set of fractions throughout."""
    first = tradeoff.iloc[0]
    assert first.ship_recall == pytest.approx(1.0, abs=1e-9)
    assert first.empty_dropped == pytest.approx(0.0, abs=1e-9)
    for col in ("ship_recall", "empty_dropped", "tiles_kept"):
        assert tradeoff[col].between(0.0, 1.0).all()


# ----------------------------------------------------------------- the claims in the report


@pytest.mark.parametrize("empty_dropped,classic_recall", CLASSIC)
def test_gate_dominates_classic_filter(tradeoff, empty_dropped, classic_recall):
    """At every operating point the classic filter was measured at, the gate keeps more ships.

    This is the headline of the WP3 report (+35 to +64 points). If a re-train regresses below
    the classic filter anywhere, this fails.
    """
    row = tradeoff.iloc[(tradeoff.empty_dropped - empty_dropped).abs().argmin()]
    assert row.ship_recall > classic_recall + 0.25, (
        f"at {row.empty_dropped:.3f} empties dropped the gate keeps {row.ship_recall:.3f}, "
        f"barely above the classic filter's {classic_recall}")


def test_gate_reaches_99_percent_recall(tradeoff):
    """The gate must be able to run at a 99% safety target at all, and save something there."""
    ok = tradeoff[tradeoff.ship_recall >= 0.99]
    assert len(ok), "no threshold reaches 99% ship recall"
    assert ok.empty_dropped.max() > 0.30, (
        "at 99% recall the gate drops less than 30% of empty tiles -- worse than reported")


def test_val_selected_threshold_generalises(val_selected):
    """Threshold picked on val must transfer to test; a big gap means it was fitted to noise."""
    for target, rec in val_selected.items():
        gap = rec["val"]["ship_recall"] - rec["test"]["ship_recall"]
        assert abs(gap) < 0.01, f"{target}: val->test recall gap {gap:.4f} is too large"
        assert rec["test"]["ship_recall"] > 0.98, f"{target}: test recall {rec['test']['ship_recall']}"


EMPTY_FRACTION = 0.204          # measured: 1,086 of 5,320 test tiles hold no ship


def test_compute_saving_cannot_exceed_the_empty_fraction(tradeoff):
    """The bound the old chart axis hid: saving compute means skipping tiles, and only 20.4%
    of tiles are empty. So at any operating point that keeps the ships, the share of detector
    calls skipped (1 - tiles_kept) cannot be much more than 0.204 -- anything beyond that is
    being paid for with ships, not won for free.
    """
    safe = tradeoff[tradeoff.ship_recall >= 0.99]
    assert len(safe), "no threshold reaches 99% ship recall"
    saving = 1.0 - safe.tiles_kept.min()
    assert saving <= EMPTY_FRACTION + 0.01, (
        f"claims {saving:.1%} of detector calls skipped at >=99% recall, but only "
        f"{EMPTY_FRACTION:.1%} of tiles are empty")


# ----------------------------------------------------------------- the model itself


def test_gate_model_is_small_and_well_formed():
    """~47k parameters, scores in (0, 1), one score per tile, and a deterministic forward pass."""
    pytest.importorskip("torch.utils.data", reason="no working torch in .venv; use .venv312")
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from wp3_train_gate import Gate, load_gate

    m = Gate().eval()
    n_par = sum(p.numel() for p in m.parameters())
    assert 10_000 < n_par < 100_000, f"{n_par} parameters is not 'small enough for a satellite'"

    x = torch.zeros(3, 3, 128, 128)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert a.shape == (3,)
    assert torch.equal(a, b), "forward pass is not deterministic in eval mode"
    assert torch.sigmoid(a).min() > 0.0 and torch.sigmoid(a).max() < 1.0

    ck = Path(__file__).resolve().parents[1] / "models" / "gate.pt"
    if ck.exists():
        loaded, px = load_gate(ck)
        assert px == 128
        assert sum(p.numel() for p in loaded.parameters()) == n_par


def test_trained_gate_separates_ships_from_empty():
    """The trained checkpoint must score a textured tile differently from a flat one.

    Not a performance claim -- a smoke test that `gate.pt` holds trained weights rather than a
    re-initialised model, which is what a broken load would silently give.
    """
    pytest.importorskip("torch.utils.data", reason="no working torch in .venv; use .venv312")
    import torch
    ck = Path(__file__).resolve().parents[1] / "models" / "gate.pt"
    if not ck.exists():
        pytest.skip("models/gate.pt missing -- run scripts/wp3_train_gate.py")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from wp3_train_gate import load_gate

    m, px = load_gate(ck)
    rng = np.random.default_rng(0)
    flat = torch.zeros(1, 3, px, px)
    noisy = torch.from_numpy(rng.random((1, 3, px, px)).astype("float32"))
    with torch.no_grad():
        s_flat = float(torch.sigmoid(m(flat)))
        s_noisy = float(torch.sigmoid(m(noisy)))
    assert abs(s_flat - s_noisy) > 1e-3, (
        f"flat {s_flat:.4f} and textured {s_noisy:.4f} score the same -- weights look untrained")
