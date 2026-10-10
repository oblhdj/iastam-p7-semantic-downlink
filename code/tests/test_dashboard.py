"""demo/dashboard.py and its two modules: the page must tell the truth and must not crash.

Pins: the canon the page reads is the committed canon (341x, B1/B2 from the same-scene run, the
six-day run, the sensitivity count, the SAHI energy variant); the stdlib arithmetic equals
sat7.accounting; the live pipeline's bytes are the real stream's; routes behave as labelled; bad
inputs raise a readable error; the stored fallback assets load and match today's canon; and the
page itself renders with no exception both live and with the live pipeline switched off.

The live tests need onnxruntime and the page tests need streamlit; each skips where its package is
absent (install demo/requirements-dashboard.txt to run them all).
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demo"
sys.path.insert(0, str(DEMO))

import demo_data as D  # noqa: E402

HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None
HAVE_ST = importlib.util.find_spec("streamlit") is not None


def live(fn):
    """A test that runs the detector on CPU: tier `inference`, skipped without onnxruntime."""
    skip = pytest.mark.skipif(not HAVE_ORT, reason="onnxruntime not installed (demo/requirements-dashboard.txt)")
    return pytest.mark.inference(skip(fn))


def page(fn):
    """A test that renders the dashboard (and so runs the detector): tiers `page` and `inference`."""
    skip = pytest.mark.skipif(not (HAVE_ST and HAVE_ORT), reason="streamlit / onnxruntime not installed")
    return pytest.mark.page(pytest.mark.inference(skip(fn)))


# ------------------------------------------------------------------------------ canon
def test_canon_is_the_committed_canonical_state():
    c = D.canon()
    assert c["missing"] == []
    h = c["headline"]
    assert round(h["reduction_x"]) == 341 and round(h["earlier_all_modeled_x"]) == 557
    assert round(h["MB_sent_per_day"]) == 176 and round(h["ship_recall"], 3) == 0.685
    assert h["measured_share_percent"] + h["modeled_share_percent"] == pytest.approx(100.0)
    rec = {r["key"]: r for r in c["detection"]["rows"]}
    assert rec["B1"]["recall"] == 0.3385 and rec["B2"]["recall"] == 0.7174
    assert rec["native"]["recall"] == 0.7658 and not rec["native"]["is_paper_baseline"]
    assert not rec["per_tile"]["is_paper_baseline"] and "NOT the paper's B1" in rec["native"]["config"]
    assert (c["sensitivity"]["holds"], c["sensitivity"]["of"]) == (74, 80)
    assert round(c["sensitivity"]["lead_at_160k_pts"], 1) == 1.3


def test_canon_six_day_run_and_single_day_overload():
    ss = D.canon()["steady_state"]
    ours, fifo = ss["runs"]["share_0.25_Value-greedy (ours)"], ss["runs"]["share_0.25_FIFO"]
    r = [d["ship_recall"] for d in ours["per_day"]]
    assert (round(min(r), 3), round(max(r), 3)) == (0.677, 0.688) and len(r) == 6
    assert ours["offered_over_sustained"] > 1.0          # a nominal day exceeds what a day sustains
    assert round(fifo["per_day"][0]["latency_med_h"]) == 9 and round(fifo["per_day"][-1]["latency_med_h"]) == 43


def test_canon_energy_default_and_labelled_sahi_variant():
    e = D.canon()["energy"]
    assert e["default"]["E_proc_kJ_day"] == 45.64 and e["sahi_variant"]["E_proc_kJ_day"] == 69.96
    assert e["sahi_variant"]["calls_per_tile"] == 1.5625 and "ESTIMATE" in e["label"]


def test_canon_reports_missing_files_instead_of_inventing(tmp_path):
    c = D.canon(tmp_path)
    assert c["headline"] is None and c["detection"] is None and c["steady_state"] is None
    assert "wp6_real_table.csv" in c["missing"] and "wp26_b0_b4.json" in c["missing"]


def test_page_source_types_in_no_canonical_number():
    src = (DEMO / "dashboard.py").read_text(encoding="utf-8")
    for literal in ("341", "557", "0.339", "0.717", "0.3385", "0.7174", "0.685", "0.677", "0.688",
                    "176", "134.5", "74 of", "42.9", "45.6", "70.0", "69.96"):
        assert literal not in src, f"dashboard.py hard-codes {literal!r}; read it from code/results"


def test_stdlib_arithmetic_equals_sat7_accounting():
    from sat7.accounting import data_reduction_percent
    for tx, raw in ((45_681, 15_925_248), (0, 1_769_472), (176.27e6, 60124.3e6)):
        assert D.data_reduction_percent(tx, raw) == pytest.approx(data_reduction_percent(tx, raw))


def test_environment_names_a_fix_for_everything_missing():
    env = D.environment()
    for r in env["rows"]:
        assert r["ok"] or r["fix"], r["name"]
    assert env["live_ok"] == (not env["blocking"])


def test_first_run_config_needs_nobody_at_the_keyboard():
    """Found by a clean-clone run: without this file `streamlit run` waits at an "Email:" prompt on
    a machine that has never run Streamlit. It must sit next to the script to apply from any cwd."""
    import tomllib
    cfg = tomllib.loads((DEMO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["server"]["showEmailPrompt"] is False
    assert cfg["server"]["address"] == "localhost"          # no listen on every interface
    assert cfg["browser"]["gatherUsageStats"] is False
    assert "headless" not in cfg["server"]                  # headless would stop the browser opening
    assert {r["name"] for r in D.environment()["rows"] if r["optional"]} == {
        "full demo: Airbus split", "full demo: torch + GPU env"}


def test_ground_truth_tile_survives_an_empty_label_file():
    """Found by hand: a real tile with no ships, uploaded with its (empty) label file, made the
    Results section fail on `None` recall. One test tile in five has an empty label file."""
    assert D.ground_truth_tile(None, 1.0)[:2] == ("Onboard detections transmitted", "100%")
    assert D.ground_truth_tile(None, None)[1] == "—"
    label, value, help_text = D.ground_truth_tile({"n_gt": 0, "recall": None, "FP": 2}, None)
    assert (label, value) == ("False alarms transmitted", "2") and "undefined" in help_text
    scored = D.ground_truth_tile({"n_gt": 7, "recall": 1.0, "FP": 0}, 1.0)
    assert scored[:2] == ("Ships reported / ground truth", "1.000")


def test_label_file_pairs_with_its_image_by_name():
    assert not D.label_name_mismatch("00113a75c.jpg", "00113a75c.txt")
    assert not D.label_name_mismatch("Tile.JPG", "tile.txt")
    assert D.label_name_mismatch("03204a586.jpg", "00113a75c.txt")       # the previous image's labels


# ------------------------------------------------------------------------------ fallback assets
def test_fallback_assets_load_and_match_todays_canon():
    index = D.fallback_index()
    assert "error" not in index, index
    assert {s["key"] for s in index["samples"]} >= {"swath"}
    assert D.fallback_staleness(index) == []
    for s in index["samples"]:
        bundle, files = D.load_fallback(s["key"])
        assert bundle["source"]["synthetic"] and bundle["source"]["label"] == "SYNTH"
        assert len(files["downlink_B3.bin"]) == bundle["modes"]["B3"]["bytes"]["total"]
        for mode in D.MODES:
            b = bundle["modes"][mode]["bytes"]
            assert b["payload"] + b["overhead"] == b["total"]
            assert D.mode_summary(bundle, mode, "policy")["bytes_raw"] == bundle["input"]["raw_bytes"]
        assert "SYNTH" in D.metrics_csv(bundle).splitlines()[1]


def test_fallback_errors_are_readable(tmp_path):
    assert "not found" in D.fallback_index(tmp_path)["error"]
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        D.load_fallback("swath", tmp_path)
    with pytest.raises(FileNotFoundError, match="no fallback sample"):
        D.load_fallback("nope")


# ------------------------------------------------------------------------------ live pipeline
@pytest.fixture(scope="module")
def swath_run():
    import demo_pipeline as P
    det, model = P.load_model()
    img, info, _ = P.bundled_swath()
    src = {"kind": "sample", "label": "SYNTH", "synthetic": True, "note": ""}
    bundle, art = P.run_all(img, info, det, P.Settings(), source=src, model_path=model)
    return P, det, bundle, art


@live
def test_live_bytes_are_the_real_stream_and_modes_differ_only_by_their_switches(swath_run):
    P, _det, b, art = swath_run
    m = b["modes"]
    assert len(art["downlink_B3.bin"]) == m["B3"]["bytes"]["total"]
    for mode in D.MODES:
        assert m[mode]["bytes"]["payload"] + m[mode]["bytes"]["overhead"] == m[mode]["bytes"]["total"]
    assert m["B0"]["bytes"]["payload"] == b["input"]["raw_bytes"] == 2304 * 2304 * 3
    assert D.mode_summary(b, "B0")["reduction_percent"] < 0          # raw + packet headers
    assert (m["B1"]["detector_calls"], m["B2"]["detector_calls"]) == (1, 16)
    assert m["B3"]["bytes"]["total"] == m["B4"]["bytes"]["total"]    # the relay changes when, not what
    assert m["B3"]["levels"] == m["B4"]["levels"]
    assert m["B2"]["n_transmitted"] == b["perception"]["sahi"]["n_at_or_above_cut"]
    assert len(m["B3"]["records"]) == m["B3"]["n_transmitted"]       # ground decode of the stream
    assert m["B3"]["quality"] is None and "no ground truth" in b["labels"]["accuracy"]
    assert b["meta"]["elapsed_s"] < 60


@live
def test_live_packet_equals_the_quickstart_packet(swath_run):
    """Same tiles, same sat7 encoder: the dashboard's B3 stream is the quickstart's downlink.bin."""
    _P, _det, b, art = swath_run
    out = DEMO / "quickstart" / "_out" / "downlink.bin"
    if not out.exists():
        pytest.skip("run demo/quickstart/run_demo.py once to write _out/downlink.bin")
    assert len(art["downlink_B3.bin"]) == out.stat().st_size


@live
def test_routes_are_what_their_labels_say(swath_run):
    _P, _det, b, _art = swath_run
    runs = b["comms"]["runs"]
    assert set(runs["B3"]) == {"direct"} and set(runs["B4"]) == {"direct", "policy", "relay"}
    d, p, r = (runs["B4"][k] for k in ("direct", "policy", "relay"))
    assert d["relay_items"] == 0 and r["direct_items"] == 0 and r["relay_items"] == r["items"]
    assert d["latency_last_s"] == pytest.approx(runs["B3"]["direct"]["latency_last_s"])
    assert p["latency_last_s"] <= d["latency_last_s"] + 1e-6         # min-J with lam_E = 0 is min-T
    assert r["energy_J"]["isl"] > 0 and r["energy_J"]["relay_ground"] > 0 and r["energy_J"]["ground"] == 0
    assert "manual" in r["chosen_by"] and "policy" in p["chosen_by"]
    s = D.mode_summary(b, "B4", "policy")
    assert s["T_total_s"] == pytest.approx(s["T_processing_s"] + s["T_encoding_s"] + s["T_communication_s"]
                                           + s["T_decoding_s"])
    assert s["E_total_J"] == pytest.approx(s["E_processing_J"] + s["E_communication_J"])
    assert len(b["comms"]["sweep"]["rows"]) == 24


@live
def test_cloud_tile_sends_nothing_and_override_is_labelled(swath_run):
    P, det, _b, _art = swath_run
    name = "09_synthetic_024e6ba29.jpg"
    img, info = P.read_image((DEMO / "quickstart" / "tiles" / name).read_bytes(), name)
    b, _ = P.run_all(img, info, det, P.Settings(sweep=False))
    assert b["grid"]["tiles"][0]["context"] == "cloud" and b["modes"]["B3"]["bytes"]["total"] == 0
    assert b["perception"]["one_window"] and b["source"]["label"] == "UPLOAD"
    forced, _ = P.run_all(img, info, det, P.Settings(context="ships", sweep=False))
    assert forced["grid"]["tiles"][0]["context"] == "ships" and "open sea" in forced["grid"]["context_source"]


@live
def test_ground_truth_turns_counts_into_metrics(swath_run):
    P, det, b, _art = swath_run
    img, info, _ = P.bundled_swath()
    top = b["perception"]["sahi"]["detections"][0]
    line = f"0 {top['cx'] / info.width} {top['cy'] / info.height} {top['w'] / info.width} {top['h'] / info.height}"
    gts = P.parse_labels(line + "\n", info)
    scored, _ = P.run_all(img, info, det, P.Settings(sweep=False), gts=gts)
    q = scored["modes"]["B2"]["quality"]
    assert q["n_gt"] == 1 and q["TP"] == 1 and q["recall"] == 1.0
    assert scored["modes"]["B0"]["quality"] is None                  # no detector, no detection metric


@live
def test_bad_inputs_raise_readable_errors():
    import cv2
    import numpy as np
    import demo_pipeline as P
    with pytest.raises(ValueError, match="empty file"):
        P.read_image(b"", "x.jpg")
    with pytest.raises(ValueError, match="not a supported image"):
        P.read_image(b"this is not an image", "x.jpg")
    with pytest.raises(ValueError, match="does not decode"):
        P.read_image(b"\xff\xd8\xff" + b"\x00" * 64, "cut.jpg")
    big = cv2.imencode(".png", np.zeros((8, P.MAX_SIDE + 1, 3), np.uint8))[1].tobytes()
    with pytest.raises(ValueError):
        P.read_image(big, "wide.png")
    _img, info = P.read_image(cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes(), "s.png")
    with pytest.raises(ValueError, match="YOLO format"):
        P.parse_labels("ship 10 20 30 40\n", info)
    with pytest.raises(FileNotFoundError, match="git checkout"):
        P.load_model(Path("no_such_model.onnx"))
    with pytest.raises(ValueError, match="route must be"):
        P.simulate_route([], "teleport")


@live
def test_an_empty_label_file_is_ground_truth_with_no_ships(swath_run):
    P, det, _b, _art = swath_run
    name = "05_synthetic_00293fb9e.jpg"                    # stands in for an open-sea tile with no ships
    img, info = P.read_image((DEMO / "quickstart" / "tiles" / name).read_bytes(), name)
    assert P.try_labels("", info) == ([], None)
    b, _ = P.run_all(img, info, det, P.Settings(sweep=False), gts=[])
    q = b["modes"]["B3"]["quality"]
    assert q["n_gt"] == 0 and q["recall"] is None and b["input"]["ground_truth"] == {"n": 0}
    tile = D.ground_truth_tile(q, b["modes"]["B3"]["transmitted_share_of_onboard"])
    assert tile[0] == "False alarms transmitted" and tile[1] == str(q["FP"])
    assert D.metrics_csv(b) and D.records_csv(b) and D.slim(b)        # the exports still build


@live
def test_a_bad_label_file_is_reported_and_the_image_is_kept():
    import cv2
    import numpy as np
    import demo_pipeline as P
    _img, info = P.read_image(cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes(), "s.png")
    for text in ("ship 10 20 30 40\n", "1 0.5 0.5 0.1 0.1\n"):
        gts, why = P.try_labels(text, info)
        assert gts is None and "YOLO format" in why
    gts, why = P.try_labels("0 0.5 0.5 0.25 0.25\n", info)
    assert why is None and gts == [(32.0, 32.0, 16.0, 16.0)]


# ------------------------------------------------------------------------------ the page
def _run_page(monkeypatch, fallback: bool):
    from streamlit.testing.v1 import AppTest
    if fallback:
        monkeypatch.setenv("P7_DEMO_FORCE_FALLBACK", "1")
    else:
        monkeypatch.delenv("P7_DEMO_FORCE_FALLBACK", raising=False)
    return AppTest.from_file(str(DEMO / "dashboard.py"), default_timeout=180).run()


@page
def test_page_renders_live_in_every_mode_and_route(monkeypatch):
    at = _run_page(monkeypatch, fallback=False)
    assert not at.exception and not at.error
    for mode in D.MODES:
        at.radio(key="mode").set_value(mode).run()
        assert not at.exception and not at.error, mode
    for route in D.ROUTES:
        at.radio(key="route").set_value(route).run()
        assert not at.exception and not at.error, route
    labels = {m.label: m.value for m in at.metric}
    c = D.canon()["headline"]
    assert labels["Data reduction, simulated day"].startswith(f"{c['reduction_x']:,.0f}")
    assert labels["Data reduction, this image"].endswith("%")                    # never mixed with the canon
    text = " ".join(x.value for x in at.caption)
    assert "MEASURED" in text and "MODELED" in text and "SYNTH" in text and "ESTIMATE" in text


@page
def test_page_renders_from_static_fallback_when_the_pipeline_is_off(monkeypatch):
    at = _run_page(monkeypatch, fallback=True)
    assert not at.exception
    assert [e.value for e in at.error] == ["The live pipeline cannot run here, so the page is showing "
                                           "pre-generated results."]
    assert any("STATIC results" in w.value for w in at.warning)
    sample = next(s for s in at.selectbox if s.label == "Sample")
    for key in [s["key"] for s in D.fallback_index()["samples"]]:
        sample.set_value(key).run()
        assert not at.exception and len(at.error) == 1, key


@page
def test_page_upload_widgets_with_label_files(monkeypatch):
    """The upload path as a presenter uses it: image, then label files good, empty, malformed and
    belonging to another image; then a file that is not an image. Synthetic tile: no Airbus pixels."""
    name = "05_synthetic_00293fb9e"
    tile = (DEMO / "quickstart" / "tiles" / f"{name}.jpg").read_bytes()
    at = _run_page(monkeypatch, fallback=False)
    next(r for r in at.radio if r.label == "Image").set_value("Upload").run()
    image = lambda: next(u for u in at.file_uploader if u.label.startswith("Satellite image"))
    labels = lambda: next(u for u in at.file_uploader if u.label.startswith("Optional ground truth"))
    metric = lambda: {m.label: m.value for m in at.metric}
    assert any("No image yet" in i.value for i in at.info) and not at.exception

    image().set_value((f"{name}.jpg", tile, "image/jpeg")).run()
    assert not at.exception and not at.error and metric()["Dimensions (px)"] == "768 × 768"

    labels().set_value((f"{name}.txt", b"", "text/plain")).run()            # no ships: valid, empty
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert "False alarms transmitted" in metric() and not any("Check the pairing" in w.value for w in at.warning)

    labels().set_value((f"{name}.txt", b"0 0.5 0.5 0.1 0.1\n", "text/plain")).run()
    assert not at.exception and not at.error and "Ships reported / ground truth" in metric()

    labels().set_value((f"{name}.txt", b"ship 10 20 30 40\n", "text/plain")).run()    # malformed
    assert [e.value for e in at.error] and all("Label file ignored" in e.value for e in at.error)
    assert "Dimensions (px)" in metric() and "Onboard detections transmitted" in metric()

    labels().set_value(("another_tile.txt", b"not yolo", "text/plain")).run()         # ignored: one message,
    assert len(at.error) == 1 and not any("Check the pairing" in w.value for w in at.warning)   # not two

    labels().set_value(("another_tile.txt", b"0 0.5 0.5 0.1 0.1\n", "text/plain")).run()
    assert not at.error and any("Check the pairing" in w.value for w in at.warning)

    image().set_value(("broken.jpg", b"this is not an image", "image/jpeg")).run()
    assert not at.exception and any("Image rejected" in e.value for e in at.error)
    assert "Dimensions (px)" not in metric()
