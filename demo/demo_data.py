"""What the dashboard can show with NOTHING installed: canon, environment report, static fallback.

Standard library only, on purpose. demo/dashboard.py imports this module first, so that a laptop
with a broken environment (no onnxruntime, no OpenCV, no weights, no wifi) still opens the same
interface on pre-generated results instead of a stack trace.

  canon()              the canonical numbers, read from code/results/*.csv|json at call time.
                       Nothing here is typed in: a number the results files do not carry is absent,
                       and its file is listed under "missing".
  environment()        what is installed / present, what each missing piece disables, how to fix it.
  load_fallback()      demo/fallback_assets/: bundles written by demo/make_fallback_assets.py from a
                       known-good live run (same structure the live pipeline returns).
  mode_summary()       per-mode bytes / reduction / latency / energy from a bundle. The arithmetic is
                       sat7.accounting's, restated without imports; tests assert they agree.

Labels are the repo's: REAL, SIM, SIM-over-REAL, ASSUMPTION, plus SYNTH for anything computed on
the bundled synthetic stand-in tiles and ESTIMATE for every joule (no power was ever measured).
"""
from __future__ import annotations

import csv
import importlib
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "code" / "results"
QUICKSTART = REPO / "demo" / "quickstart"
FALLBACK = REPO / "demo" / "fallback_assets"
MODEL_CANDIDATES = (QUICKSTART / "model" / "best.onnx",          # tracked in git: the demo's model
                    REPO / "code" / "runs" / "ships" / "weights" / "best.onnx")
BUNDLE_VERSION = 1
MODES = ("B0", "B1", "B2", "B3", "B4")
CONTEXTS = {"prefilter": "pre-filter decides (default)", "ships": "treat every tile as open sea",
            "coast": "treat every tile as coast"}       # the last two are manual overrides
ROUTES = {"direct": "direct to the ground station",
          "policy": "policy-chosen (min J = lam_E x E + lam_T x T per item)",
          "relay": "relay only (manual)"}

MODE_INFO = {
    "B0": {"title": "B0 — raw downlink",
           "perception": "none", "payload": "the whole image, uncompressed", "relay": False,
           "adds": "Nothing onboard. The baseline: every pixel goes down (H x W x 3 bytes, behind "
                   "an 18 B image header, in CCSDS packets)."},
    "B1": {"title": "B1 — plain YOLO",
           "perception": "one detector call on the whole image", "payload": "detection records",
           "relay": False,
           "adds": "A detector onboard. One call on the original image, however large (the 768 px "
                   "letterbox shrinks a big scene), then one 34 B record per detection at or above "
                   "the onboard cut. No pictures."},
    "B2": {"title": "B2 — SAHI + YOLO",
           "perception": "SAHI windows + global-coordinate fusion", "payload": "detection records",
           "relay": False,
           "adds": "Slicing. The image is cut into overlapping 768 px windows, each detected at "
                   "native resolution, boxes lifted to image coordinates and duplicates fused. "
                   "Costs more detector calls; still records only."},
    "B3": {"title": "B3 — semantic downlink",
           "perception": "SAHI windows + fusion", "payload": "P0-P3 semantic packet",
           "relay": False,
           "adds": "The semantic policy (paper Table I). Each detection gets a level from its "
                   "confidence and its tile's context: P1 record only, P2 + ROI crop when "
                   "uncertain, P3 + context image on a coast; cloud tiles are dropped (P0)."},
    "B4": {"title": "B4 — semantic + relay",
           "perception": "SAHI windows + fusion", "payload": "P0-P3 semantic packet",
           "relay": True,
           "adds": "A second way down. The same packet as B3; each item may go direct, or over "
                   "an inter-satellite link to a relay that forwards it on its own ground pass. "
                   "Changes when data lands and what it costs to send, never what is sent."},
}


# ====================================================================================== readers
def _json(name: str, results: Path = RESULTS):
    p = results / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _table(name: str, results: Path = RESULTS) -> list[dict]:
    p = results / name
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict], strategy: str) -> dict | None:
    return next((r for r in rows if r.get("strategy") == strategy), None)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ====================================================================================== canon
def canon(results: Path = RESULTS) -> dict:
    """The canonical results, each with the label and file it has in the repo. Sections whose file
    is absent are None and the file is named in ["missing"]."""
    out: dict = {"missing": []}

    def need(name, loader=_json):
        v = loader(name, results)
        if not v:
            out["missing"].append(name)
        return v

    # ---- validated ledger (wp29): one place that already ties each headline number to its file
    w29 = need("wp29_validation.json")
    ledger = {c["id"]: c for c in (w29 or {}).get("claims_ledger", [])}
    led = lambda k: ledger.get(k, {}).get("value")
    out["ledger"] = {k: {"value": c["value"], "source": c["source"]} for k, c in ledger.items()}

    # ---- headline: the simulated day
    table = need("wp6_real_table.csv", _table)
    ours, bent = _row(table, "Ours: LoD + value-greedy"), _row(table, "Bent pipe (raw, FIFO)")
    fair, fifo = _row(table, "Phi-sat-2 style, fair (+coastal)"), _row(table, "Ours: LoD + FIFO")
    old = _row(_table("wp6_real_table_modeledcoast.csv", results), "Ours: LoD + value-greedy")
    out["headline"] = None
    if ours and bent:
        sent, raw = _f(ours["MB_sent"]), _f(bent["MB_offered"])
        modeled = led("modeled_share_of_bytes_percent")
        out["headline"] = {
            "reduction_x": _f(ours["data_reduction_x"]),
            "reduction_percent": 100.0 * (1.0 - sent / raw),
            "MB_sent_per_day": sent, "MB_raw_per_day": raw,
            "bent_pipe_MB_sent_per_day": _f(bent["MB_sent"]),
            "ship_recall": _f(ours["ship_recall"]), "cloud_band": led("cloud_band"),
            "latency_med_h": _f(ours["latency_med_h"]),
            "fifo_latency_med_h": _f(fifo["latency_med_h"]) if fifo else None,
            "fair_baseline_recall": _f(fair["ship_recall"]) if fair else None,
            "modeled_share_percent": modeled,
            "measured_share_percent": None if modeled is None else 100.0 - modeled,
            "earlier_all_modeled_x": _f(old["data_reduction_x"]) if old else None,
            "label": "SIM-over-REAL",
            "measured": "coastal context tiles: real JPEG serializations of all 1,415 coastal "
                        "test tiles (wp28)",
            "modeled": "thumbnails and ship crops: model sizes (flat 1,000 B, WP4 power law)",
            "source": "code/results/wp6_real_table.csv, row 'Ours: LoD + value-greedy'",
        }

    # ---- detection: which recall belongs to which configuration
    w26, w24 = need("wp26_b0_b4.json"), need("wp24_detection_eval.json")
    out["detection"] = None
    if w26:
        q = w26["detection_quality_same_scenes"]
        inp = w26["inputs"]
        scope = (f"{inp['scenes']} scenes of {inp['per_side']}x{inp['per_side']} real tiles "
                 f"({inp['scene_px']} px), {inp['ships']} ships")
        rows = [
            {"key": "B1", "config": "B1 — one detector call on the whole scene (no slicing)",
             "is_paper_baseline": True, "scope": scope, "source": "wp26_b0_b4.json", **_pick(q["B1"])},
            {"key": "B2", "config": "B2 — SAHI + fusion on the same scenes",
             "is_paper_baseline": True, "scope": scope, "source": "wp26_b0_b4.json", **_pick(q["B2"])},
            {"key": "B3", "config": "B3/B4 — what the semantic packet still reports (cloud tiles dropped)",
             "is_paper_baseline": True, "scope": scope, "source": "wp26_b0_b4.json", **_pick(q["B3"])},
            {"key": "per_tile", "config": "reference — one call per native 768 px tile, same scenes "
                                          "(a 16-window tiling: NOT the paper's B1)",
             "is_paper_baseline": False, "scope": scope, "source": "wp26_b0_b4.json",
             **_pick(q["B1_per_tile_reference"])},
        ]
        if w24:
            n = w24["B1_committed_all_test_tiles"]
            rows.append({"key": "native", "config": "reference — detector on native tiles, full test "
                                                    "split (NOT the paper's B1)",
                         "is_paper_baseline": False,
                         "scope": f"{n['images']} tiles, {n['n_gt']} ships",
                         "source": "wp24_detection_eval.json", **_pick(n)})
        out["detection"] = {"rows": rows, "label": "REAL", "conf_thr": q["B1"]["conf_thr"],
                            "iou_thr": q["B1"]["iou_thr"],
                            "sahi_compute_x": led("sahi_compute_x"),
                            "windows_per_scene": w26["checks"]["3_B2_sahi_fusion"]["windows_per_scene"]}
        out["campaign"] = {"label": w26["label"], "day": w26["day"]["results"],
                           "mean_bytes_per_scene": w26["mean_bytes_per_scene"],
                           "tiles": w26["day"]["tiles"], "ships": w26["day"]["ships"],
                           "source": "code/results/wp26_b0_b4.json"}
    else:
        out["campaign"] = None

    # ---- communication: direct vs relay (SIM)
    w27 = need("wp27_comms.json")
    out["comms"] = None
    if w27:
        out["comms"] = {"label": w27["label"], "config": w27["config"],
                        "contact_plan": w27["contact_plan"],
                        "nominal": {k: w27["nominal"][k] for k in ("direct_only", "with_relay")},
                        "congested": {k: w27["congested"][k] for k in ("direct_only", "with_relay")},
                        "tiles": {"nominal": w27["nominal"]["tiles"], "congested": w27["congested"]["tiles"]},
                        "source": "code/results/wp27_comms.json"}

    # ---- the six-day steady state (the single-day caveat)
    out["steady_state"] = None
    if w29 and "steady_state" in w29:
        ss = w29["steady_state"]
        runs = ss["runs"]
        out["steady_state"] = {
            "label": ss["label"], "days": ss["days"], "tiles_per_day": ss["tiles_per_day"],
            "offered_MB_per_day_mean": ss["offered_MB_per_day_mean"],
            "runs": {k: {"policy": v["policy"], "link_share": v["link_share"],
                         "sustained_capacity_MB_per_24h": v["sustained_capacity_MB_per_24h"],
                         "offered_over_sustained": v["offered_over_sustained"],
                         "per_day": v["per_day"]} for k, v in runs.items()},
            "source": "code/results/wp29_validation.json, steady_state",
        }

    # ---- energy (every joule an ESTIMATE: powers assumed)
    w17 = need("wp17_energy_model.json")
    out["energy"] = None
    if w17:
        day = w17["E_per_day_kJ"]
        var = w17.get("sahi_variant")
        out["energy"] = {
            "label": "ESTIMATE — stage times REAL (laptop, not flight hardware), link SIM, "
                     "every power an ASSUMPTION",
            "powers_W": w17["powers_W_assumption"], "stage_times_ms": w17["stage_times_ms"],
            "default": {"what": "one detector call per tile (what the simulated day runs)",
                        "E_proc_kJ_day": day["E_proc_proposed"], "E_comm_kJ_day": day["E_comm_proposed_semantic"],
                        "E_total_kJ_day": day["E_total_proposed"],
                        "proc_to_comm": day["proc_vs_comm_ratio_proposed"],
                        "ES_proc": w17["E_proc_per_tile_J"]["cpu_onnx_DEFAULT"]["ES_proc"],
                        "J_per_tile": w17["E_proc_per_tile_J"]["cpu_onnx_DEFAULT"]["E_prop"]},
            "sahi_variant": None if not var else {
                "what": var["label"], "calls_per_tile": var["detector_calls_per_tile"],
                "E_proc_kJ_day": var["E_per_day_kJ"]["E_proc_proposed"],
                "E_comm_kJ_day": var["E_per_day_kJ"]["E_comm_proposed_semantic"],
                "E_total_kJ_day": var["E_per_day_kJ"]["E_total_proposed"],
                "proc_to_comm": var["E_per_day_kJ"]["proc_vs_comm_ratio_proposed"],
                "ES_proc": var["E_proc_per_tile_J"]["ES_proc"],
                "J_per_tile": var["E_proc_per_tile_J"]["E_prop"]},
            "bent_pipe_all_raw_kJ_day": day["E_comm_bent_pipe_raw"],
            "bent_pipe_link_limited_kJ_day": led("link_limited_bent_pipe_kJ_per_day"),
            "bent_pipe_days_of_contact_needed": w17["downlink"]["bent_pipe_days_of_contact_needed"],
            "source": "code/results/wp17_energy_model.json",
        }

    # ---- sensitivity and validation status
    sens = led("sensitivity_ours_ge_fair")
    out["sensitivity"] = None if not sens else {
        "holds": sens[0], "of": sens[1], "lead_at_160k_pts": led("lead_over_fair_at_160k_pts"),
        "lead_nominal_pts": led("lead_over_fair_baseline_pts"),
        "label": "SIM-over-REAL", "source": "code/results/wp10_sensitivity.csv (via the wp29 ledger)"}
    out["validation"] = None if not w29 else {
        k: {"title": v["title"], "status": v["status"], "limits": v.get("limits", [])}
        for k, v in w29.get("checklist", {}).items()}
    w25 = _json("wp25_semantic_packets.json", results)
    out["stage_timing"] = None if not w25 else w25.get("timing_ms_per_tile")
    return out


def _pick(m: dict) -> dict:
    return {k: m.get(k) for k in ("precision", "recall", "F1", "AP50", "n_gt", "TP", "FP", "FN")}


def canon_snapshot(c: dict | None = None) -> dict:
    """The handful of numbers a fallback bundle is stamped with, so a stale bundle can be spotted."""
    c = c or canon()
    h, d, s, st = c.get("headline"), c.get("detection"), c.get("sensitivity"), c.get("steady_state")
    rec = {r["key"]: r["recall"] for r in d["rows"]} if d else {}
    snap = {"reduction_x": round(h["reduction_x"], 1) if h else None,
            "MB_sent_per_day": round(h["MB_sent_per_day"], 1) if h else None,
            "B1_recall": rec.get("B1"), "B2_recall": rec.get("B2"),
            "sensitivity": [s["holds"], s["of"]] if s else None}
    if st:
        ours = st["runs"].get("share_0.25_Value-greedy (ours)")
        fifo = st["runs"].get("share_0.25_FIFO")
        if ours:
            r = [x["ship_recall"] for x in ours["per_day"]]
            snap["six_day_recall"] = [min(r), max(r)]
        if fifo:
            snap["fifo_latency_h"] = [fifo["per_day"][0]["latency_med_h"], fifo["per_day"][-1]["latency_med_h"]]
    return snap


# ====================================================================================== environment
def _try_import(module: str):
    try:
        m = importlib.import_module(module)
        return True, str(getattr(m, "__version__", "installed"))
    except Exception as e:                       # a broken wheel raises more than ImportError
        return False, f"{type(e).__name__}: {e}"


def find_model() -> Path | None:
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


def environment() -> dict:
    """Every dependency, file and dataset the demo can use, with what its absence disables and the
    command that fixes it. `live_ok` says whether the live pipeline can run at all."""
    pip = "python -m pip install -r demo/requirements-dashboard.txt"
    rows = []

    def add(name, ok, detail, needed_for, fix, required=True, optional=False):
        rows.append({"name": name, "ok": bool(ok), "detail": detail, "needed_for": needed_for,
                     "fix": "" if ok else fix, "required_for_live": required, "optional": optional})

    add("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version(), "everything",
        "install Python 3.10 or newer")
    for mod, why in (("numpy", "everything live"), ("cv2", "image decoding, pre-filter, JPEG crops"),
                     ("onnxruntime", "running the detector on CPU"),
                     ("skyfield", "orbit / contact windows (sat7.scheduler imports sat7.orbit)"),
                     ("sgp4", "orbit propagation")):
        ok, detail = _try_import(mod)
        add(f"python package: {mod}", ok, detail, why, pip)
    model = find_model()
    add("detector weights (ONNX)", model is not None,
        f"{model.relative_to(REPO)} ({model.stat().st_size / 1e6:.1f} MB)" if model else
        "none of: " + ", ".join(str(p.relative_to(REPO)) for p in MODEL_CANDIDATES),
        "any live detection",
        "demo/quickstart/model/best.onnx is tracked in git: `git checkout -- demo/quickstart/model/best.onnx`"
        " (or re-export from code/runs/ships/weights/best.pt with code/scripts/wp1_export_int8.py)")
    tiles = sorted((QUICKSTART / "tiles").glob("*.jpg"))
    add("bundled sample tiles", len(tiles) >= 9 and (QUICKSTART / "tiles" / "manifest.json").exists(),
        f"{len(tiles)} synthetic stand-in tiles in demo/quickstart/tiles", "the bundled samples "
        "(uploads still work without them)",
        "`python demo/quickstart/make_synthetic_tiles.py` regenerates them", required=False)
    c = canon()
    add("committed results (code/results)", not c["missing"],
        "all present" if not c["missing"] else "missing: " + ", ".join(c["missing"]),
        "every canonical number on the page (341x, B1/B2, six-day run, energy)",
        "`git checkout -- code/results` (they are tracked; nothing is recomputed by the demo)",
        required=False)
    fb = fallback_index()
    add("static fallback assets", bool(fb.get("samples")),
        f"{len(fb.get('samples', []))} pre-generated sample(s), written {fb.get('generated_at', '?')}"
        if fb.get("samples") else fb.get("error", "demo/fallback_assets/manifest.json not found"),
        "showing results when the live pipeline cannot run",
        "`python demo/make_fallback_assets.py` (needs the live pipeline once)", required=False)
    # ---- only the full, detector-in-the-loop demo needs these
    data = REPO / "code" / "data" / "yolo_ships"
    add("full demo: Airbus split", data.exists(), str(data.relative_to(REPO)) + (" present" if data.exists() else " absent"),
        "`bash demo/run_demo.sh` and `run_demo.py --data` (real tiles). NOT needed for this page",
        "Kaggle 'Airbus Ship Detection Challenge' zip (~30 GB) -> code/scripts/wp0_build_dataset.py",
        required=False, optional=True)
    add("full demo: torch + GPU env", (REPO / "code" / ".venv312").exists(),
        "code/.venv312 present" if (REPO / "code" / ".venv312").exists() else "code/.venv312 absent",
        "`bash demo/run_demo.sh` (wp18 / wp11 on the real split). NOT needed for this page",
        "a Python 3.12 env with torch + ultralytics (versions in audit/REPO_AUDIT.md section 6)",
        required=False, optional=True)
    live_ok = all(r["ok"] for r in rows if r["required_for_live"])
    return {"rows": rows, "live_ok": live_ok,
            "blocking": [r for r in rows if r["required_for_live"] and not r["ok"]]}


# ====================================================================================== fallback
def fallback_index(root: Path = FALLBACK) -> dict:
    p = root / "manifest.json"
    if not p.exists():
        return {"error": f"{p.relative_to(REPO) if p.is_relative_to(REPO) else p} not found"}
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": f"manifest.json unreadable: {e}"}
    if m.get("bundle_version") != BUNDLE_VERSION:
        return {"error": f"fallback assets are bundle version {m.get('bundle_version')}, this code "
                         f"reads version {BUNDLE_VERSION}: rerun demo/make_fallback_assets.py"}
    return m


def load_fallback(key: str, root: Path = FALLBACK) -> tuple[dict, dict[str, bytes]]:
    """One pre-generated sample: (bundle, {artifact name: bytes}). Raises FileNotFoundError /
    ValueError with a message fit to show the presenter."""
    index = fallback_index(root)
    if "error" in index:
        raise FileNotFoundError(index["error"])
    entry = next((s for s in index["samples"] if s["key"] == key), None)
    if entry is None:
        raise FileNotFoundError(f"no fallback sample {key!r}; available: "
                                f"{[s['key'] for s in index['samples']]}")
    d = root / key
    bundle = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    files = {}
    for name in entry["files"]:
        f = d / name
        if not f.exists():
            raise FileNotFoundError(f"fallback file missing: demo/fallback_assets/{key}/{name} "
                                    f"(rerun demo/make_fallback_assets.py)")
        files[name] = f.read_bytes()
    return bundle, files


def fallback_staleness(index: dict, c: dict | None = None) -> list[str]:
    """Differences between the canon the fallback was generated beside and today's canon."""
    now, then = canon_snapshot(c), index.get("canon_at_generation", {})
    return [f"{k}: fallback {then.get(k)!r}, current results {v!r}" for k, v in now.items()
            if then.get(k) != v]


# ====================================================================================== arithmetic
def data_reduction_percent(d_tx: float, d_raw: float) -> float:
    """100 x (1 - D_transmitted / D_raw): sat7.accounting.data_reduction_percent."""
    return 100.0 * (1.0 - d_tx / d_raw)


def ground_truth_tile(quality: dict | None, share: float | None) -> tuple[str, str, str]:
    """(label, value, help text) of the information-preservation tile for one mode.

    quality   sat7.evaluation's result for the transmitted detections, or None without labels
    share     transmitted / onboard detections at or above the cut (None when there are none)
    An EMPTY label file is valid ground truth -- a tile with no ships, which is one test tile in
    five -- and leaves recall undefined (None): the tile then counts false alarms instead.
    """
    if quality is None:
        return ("Onboard detections transmitted", "—" if share is None else f"{100 * share:.0f}%",
                "Share of what the detector found at or above the cut that is sent. NOT accuracy: "
                "without ground truth, information preservation against reality cannot be stated.")
    if not quality["n_gt"] or quality["recall"] is None:
        return ("False alarms transmitted", str(quality["FP"]),
                "The label file lists no ships for this image, so recall is undefined. Every "
                "detection transmitted here is a false alarm.")
    return ("Ships reported / ground truth", f"{quality['recall']:.3f}",
            "Recall of the transmitted records against the supplied labels (IoU 0.5).")


def label_name_mismatch(image_name: str, label_name: str) -> bool:
    """A YOLO label file carries its image's name (00113a75c.jpg <-> 00113a75c.txt). True when the
    two uploaded names do not pair up: the usual sign that the previous image's labels were left in
    place, which would score this image against another image's ships."""
    return Path(image_name).stem.lower() != Path(label_name).stem.lower()


def route_for(mode: str, b4_route: str) -> str:
    """B0-B3 have no relay: always direct. B4 takes the chosen route."""
    return b4_route if MODE_INFO[mode]["relay"] else "direct"


def mode_summary(bundle: dict, mode: str, b4_route: str = "policy") -> dict:
    """Everything the results dashboard states about one mode of one bundle, with units in the
    keys. Communication figures are None when the simulated link could not be built."""
    m = bundle["modes"][mode]
    raw = bundle["input"]["raw_bytes"]
    a = bundle["assumptions"]
    route = route_for(mode, b4_route)
    run = ((bundle.get("comms") or {}).get("runs") or {}).get(mode, {}).get(route)
    t = m["timing_s"]
    t_proc, t_enc, t_dec = t["processing"], t["encoding"], t["decoding"] or 0.0
    e_proc = a["p_cpu_W"] * (t_proc + t_enc)
    s = {"mode": mode, "route": route, "bytes_tx": m["bytes"]["total"], "bytes_raw": raw,
         "reduction_percent": data_reduction_percent(m["bytes"]["total"], raw),
         "reduction_x": raw / m["bytes"]["total"] if m["bytes"]["total"] else None,
         "detections_transmitted": m["n_transmitted"], "detector_calls": m["detector_calls"],
         "T_processing_s": t_proc, "T_encoding_s": t_enc, "T_decoding_s": t["decoding"],
         "E_processing_J": e_proc, "T_communication_s": None, "T_first_report_s": None,
         "T_total_s": None, "E_communication_J": None, "E_total_J": None, "tx_time_s": None,
         "delivered_items": None, "items": m["n_items"], "relay_items": None}
    if run:
        s["delivered_items"], s["relay_items"] = run["delivered_items"], run["relay_items"]
        s["tx_time_s"], s["E_communication_J"] = run["tx_s_total"], run["energy_J_total"]
        s["E_total_J"] = e_proc + run["energy_J_total"]
        if run["delivered_items"] == m["n_items"] and m["n_items"]:
            s["T_communication_s"] = run["latency_last_s"]
            s["T_first_report_s"] = run["latency_first_s"]
            s["T_total_s"] = t_proc + t_enc + run["latency_last_s"] + t_dec
    return s


# ====================================================================================== export
def detections_csv(bundle: dict) -> str:
    """Every detection of both perception configurations, one row each."""
    rows = ["configuration,cx,cy,w,h,confidence,at_or_above_cut,class"]
    for key, name in (("whole", "plain_yolo"), ("sahi", "sahi_fusion")):
        for d in bundle["perception"][key]["detections"]:
            rows.append(f"{name},{d['cx']},{d['cy']},{d['w']},{d['h']},{d['confidence']},"
                        f"{int(d['at_or_above_cut'])},{d['class']}")
    return "\n".join(rows) + "\n"


def records_csv(bundle: dict) -> str:
    """The semantic records each mode put on the downlink, as the ground decoded them."""
    cols = ["image_id", "class", "confidence", "cx", "cy", "w", "h", "level", "coastal", "roi_id", "context_id"]
    rows = [",".join(["mode"] + cols)]
    for mode, m in bundle["modes"].items():
        for r in m["records"]:
            rows.append(",".join([mode] + ["" if r[c] is None else str(r[c]) for c in cols]))
    return "\n".join(rows) + "\n"


METRIC_COLUMNS = ("mode", "route", "bytes_raw", "bytes_tx", "reduction_percent", "detector_calls",
                  "detections_transmitted", "T_processing_s", "T_encoding_s", "T_communication_s",
                  "T_decoding_s", "T_total_s", "tx_time_s", "E_processing_J", "E_communication_J",
                  "E_total_J", "items", "delivered_items", "relay_items")


def metrics_csv(bundle: dict, b4_route: str = "policy") -> str:
    """The results table: one row per mode, units in the column names, labels in the last column."""
    rows = [",".join(METRIC_COLUMNS + ("labels",))]
    lab = "bytes MEASURED; times MEASURED (this machine); link SIM; energy ESTIMATE (powers assumed)"
    if bundle["source"].get("synthetic"):
        lab = "SYNTH image (not a measurement); " + lab
    for mode in MODES:
        s = mode_summary(bundle, mode, b4_route)
        rows.append(",".join("" if s[c] is None else (f"{s[c]:.6g}" if isinstance(s[c], float) else str(s[c]))
                             for c in METRIC_COLUMNS) + f',"{lab}"')
    return "\n".join(rows) + "\n"


def slim(bundle: dict) -> dict:
    """The bundle without embedded JPEG crops (for a readable JSON export)."""
    b = json.loads(json.dumps(bundle))
    for m in b["modes"].values():
        for k in ("rois", "contexts"):
            for r in m[k]:
                r.pop("jpeg_b64", None)
    return b
