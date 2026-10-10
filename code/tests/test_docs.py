"""The reproduction, configuration, limitations, audit-update and live-demo documents must agree
with the repository they describe.

Nothing here recomputes a result. Canonical values are read from code/results (through
demo/demo_data.canon, the reader the dashboard uses), formatted the way the documents write them,
and looked for in the text: a document that drifts from a result file fails here. The configuration
tables are regenerated from the dataclasses and compared; the on-screen values of the live-demo
script are compared with the stored known-good run; the launcher and the simulations script are
checked against what they claim to start.
"""
import configparser
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "demo"))
sys.path.insert(0, str(REPO / "code" / "scripts"))

import demo_data as D  # noqa: E402
import launch  # noqa: E402

LIMITATIONS = REPO / "docs" / "LIMITATIONS.md"
REPRODUCE = REPO / "docs" / "REPRODUCE.md"
CONFIGURATION = REPO / "docs" / "CONFIGURATION.md"
AUDIT_UPDATE = REPO / "audit" / "AUDIT_UPDATE_2026-10-10.md"
SCRIPT = REPO / "demo" / "LIVE_DEMO_SCRIPT.md"


def _text(path: Path) -> str:
    """The document as one line of single-spaced text, so a phrase may wrap anywhere."""
    return " ".join(path.read_text(encoding="utf-8").split())


def _canon() -> dict:
    c = D.canon()
    assert c["missing"] == [], c["missing"]
    return c


def _assert_all(path: Path, expected: dict[str, str]) -> None:
    text = _text(path)
    absent = {what: s for what, s in expected.items() if s not in text}
    assert not absent, f"{path.name} does not carry: {absent}"


# ------------------------------------------------------------------------------ numbers
def test_limitations_page_quotes_the_result_files():
    c = _canon()
    h, e, s = c["headline"], c["energy"], c["sensitivity"]
    ours = c["steady_state"]["runs"]["share_0.25_Value-greedy (ours)"]
    fifo = c["steady_state"]["runs"]["share_0.25_FIFO"]
    rec = {r["key"]: r["recall"] for r in c["detection"]["rows"]}
    days = ours["per_day"]
    r = [d["ship_recall"] for d in days]
    lat = [d["latency_med_h"] for d in days]
    d, v = e["default"], e["sahi_variant"]
    _assert_all(LIMITATIONS, {
        "reduction": f"{h['reduction_x']:.0f}×",
        "earlier estimate": f"{h['earlier_all_modeled_x']:.0f}×",
        "bytes per day": f"{h['MB_raw_per_day']:,.0f} MB → {h['MB_sent_per_day']:.1f} MB",
        "modeled share": f"**{h['modeled_share_percent']:.0f} %**",
        "measured share": f"{h['measured_share_percent']:.0f} %",
        "sustained capacity": f"{ours['sustained_capacity_MB_per_24h']} MB per 24 h",
        "offer over capacity": f"{100 * ours['offered_over_sustained']:.0f} %",
        "six-day recall": f"{min(r):.3f}–{max(r):.3f}",
        "six-day latency": f"{min(lat):.1f}–{max(lat):.1f} h",
        "bytes delivered on day six": f"{100 * days[-1]['bytes_delivered_fraction']:.0f} % of the bytes",
        "FIFO latency": f"{fifo['per_day'][0]['latency_med_h']:.1f} h → {fifo['per_day'][-1]['latency_med_h']:.1f} h",
        "FIFO day-six recall": f"{fifo['per_day'][-1]['ship_recall']:.3f}",
        "processing energy": f"{d['E_proc_kJ_day']:.1f} kJ",
        "transmission energy": f"{d['E_comm_kJ_day']:.1f} kJ",
        "total energy": f"{d['E_total_kJ_day']:.1f} kJ",
        "ES_proc": f"{100 * d['ES_proc']:.1f} %",
        "bent pipe, link-limited": f"{e['bent_pipe_link_limited_kJ_day']:.1f} kJ/day",
        "bent pipe, days of contact": f"{e['bent_pipe_days_of_contact_needed']}",
        "sensitivity": f"**{s['holds']} of {s['of']}**",
        "lead at 160k": f"{s['lead_at_160k_pts']:.1f} points",
        "lead at 40k": f"{s['lead_nominal_pts']:.1f} points",
        "SAHI calls per tile": f"{v['calls_per_tile']}",
        "SAHI processing": f"{v['E_proc_kJ_day']:.1f} kJ",
        "SAHI ES_proc": f"{100 * v['ES_proc']:.1f} %",
        "SAHI compute : radio": f"{v['proc_to_comm']:.1f} : 1",
        "default compute : radio": f"{d['proc_to_comm']:.1f} : 1",
        "B1": f"{rec['B1']:.3f}", "B2": f"{rec['B2']:.3f}", "plain tiling": f"{rec['per_tile']:.3f}",
        "recall": f"{h['ship_recall']:.3f}",
        "cloud band": f"{h['cloud_band'][0]:.2f}–{h['cloud_band'][1]:.2f}",
    })
    for heading in ("One day fits; consecutive days do not", "24 % of the transmitted bytes",
                    "No power was ever measured", "six of 80 swept settings", "SAHI is a labelled variant"):
        assert heading in _text(LIMITATIONS), heading


def test_live_demo_script_says_the_canonical_numbers():
    c = _canon()
    h, s = c["headline"], c["sensitivity"]
    rec = {r["key"]: r["recall"] for r in c["detection"]["rows"]}
    ours = c["steady_state"]["runs"]["share_0.25_Value-greedy (ours)"]
    fifo = c["steady_state"]["runs"]["share_0.25_FIFO"]["per_day"]
    r = [d["ship_recall"] for d in ours["per_day"]]
    words = {9: "nine", 43: "forty-three"}
    _assert_all(SCRIPT, {
        "reduction (table)": f"{h['reduction_x']:.0f}×",
        "reduction (spoken)": f"{h['reduction_x']:.0f} times less data",
        "B1": f"finds {rec['B1']:.3f} of the ships",
        "B2": f"finds {rec['B2']:.3f}",
        "the negative result": f"{rec['B2']:.3f} against {rec['per_tile']:.3f}",
        "six-day recall": f"{min(r):.3f} to {max(r):.3f}",
        "day over capacity": f"{100 * ours['offered_over_sustained']:.0f} percent",
        "FIFO latency": f"from {words[round(fifo[0]['latency_med_h'])]} hours behind to {words[round(fifo[-1]['latency_med_h'])]}",
        "sensitivity": f"{s['holds']} of {s['of']} settings",
    })
    quarter = h["modeled_share_percent"]
    assert 20 <= quarter <= 30 and "a quarter are still model sizes" in _text(SCRIPT)


def test_live_demo_script_matches_what_the_page_shows():
    """The on-screen values in the script are those of the stored known-good run of the swath."""
    bundle, _ = D.load_fallback("swath")
    p, m = bundle["perception"], bundle["modes"]
    hist = m["B3"]["level_histogram"]
    direct = D.mode_summary(bundle, "B3", "direct")["T_communication_s"] / 3600
    b4 = D.mode_summary(bundle, "B4", "policy")
    assert bundle["settings"]["capture_h"] == 2.0 and b4["relay_items"] == b4["items"]
    _assert_all(SCRIPT, {
        "image size": f"{bundle['input']['width']} × {bundle['input']['height']} px",
        "raw size": f"{bundle['input']['raw_bytes'] / 1e6:.2f} MB",
        "plain YOLO": f"plain YOLO, {p['whole']['n_at_or_above_cut']} detections, {p['whole']['calls']} detector call",
        "SAHI": f"SAHI, {p['sahi']['n_at_or_above_cut']} detections, {p['sahi']['calls']} detector calls",
        "levels": f"{hist['P1-meta']} at P1, {hist['P2-roi']} at P2, {hist['P3-context']} at P3",
        "packet": f"packet {m['B3']['bytes']['total'] / 1e3:.1f} kB",
        "relay latency (tile)": f"{b4['T_communication_s'] / 3600:.2f} h",
        "relay vs direct (spoken)": f"in {b4['T_communication_s'] / 3600:.1f} hours, against {direct:.1f} direct",
        "capture time": "capture time 2.00",
    })


def test_audit_update_quotes_the_result_files():
    c = _canon()
    h, e = c["headline"], c["energy"]
    rec = {r["key"]: r["recall"] for r in c["detection"]["rows"]}
    ours = c["steady_state"]["runs"]["share_0.25_Value-greedy (ours)"]
    fifo = c["steady_state"]["runs"]["share_0.25_FIFO"]["per_day"]
    r = [d["ship_recall"] for d in ours["per_day"]]
    flips = json.loads((D.RESULTS / "wp11_integration.json").read_text(encoding="utf-8"))
    flips = flips["catalogue_cross_check"]["decision_flips"]
    _assert_all(AUDIT_UPDATE, {
        "B1 and B2": f"{rec['B1']:.3f} and {rec['B2']:.3f}",
        "native tiles": f"{rec['native']:.3f}",
        "plain tiling": f"{rec['B2']:.3f} vs {rec['per_tile']:.3f}",
        "headline": f"{h['earlier_all_modeled_x']:.0f}× → {h['reduction_x']:.0f}×",
        "capacity": f"{ours['sustained_capacity_MB_per_24h']} MB",
        "six-day recall": f"{min(r):.3f}–{max(r):.3f}",
        "FIFO latency": f"{fifo[0]['latency_med_h']:.1f} h → {fifo[-1]['latency_med_h']:.1f} h",
        "SAHI variant": f"{e['sahi_variant']['E_proc_kJ_day']:.1f} kJ/day of processing, ES_proc {100 * e['sahi_variant']['ES_proc']:.1f} %",
        "link-limited bent pipe": f"{e['bent_pipe_link_limited_kJ_day']:.1f} kJ/day",
        "decision flips": f"`det_thr` {flips['det_thr']}, `conf_low` {flips['conf_low']}, `conf_high` {flips['conf_high']}",
        "sensitivity": f"in {c['sensitivity']['holds']}",
    })


def test_reproduction_guide_restates_no_headline_number():
    text = _text(REPRODUCE)
    c = _canon()
    for forbidden in (f"{c['headline']['reduction_x']:.0f}×", "0.685", "0.339", "0.717"):
        assert forbidden not in text, f"REPRODUCE.md quotes {forbidden}; it should point at the results"
    assert f"{c['ledger']['dataset_tiles']['value']:,} tiles" in text


# ------------------------------------------------------------------------------ configuration
def test_configuration_page_is_generated_from_the_code():
    pytest.importorskip("cv2")                       # demo_pipeline.Settings is one of the tables
    import print_config
    text = CONFIGURATION.read_text(encoding="utf-8")
    assert print_config.BEGIN in text and print_config.END in text
    inside = text.split(print_config.BEGIN, 1)[1].split(print_config.END, 1)[0]
    assert inside.strip() == print_config.render().strip(), \
        "docs/CONFIGURATION.md is out of date: python code/scripts/print_config.py --write docs/CONFIGURATION.md"


def test_configuration_page_states_the_operating_values():
    from sat7.perception import DEFAULT_RAW_CONF
    from sat7.priority import PriorityConfig
    from sat7.scheduler import LoDConfig
    from sat7.semantic import EncoderConfig
    text = _text(CONFIGURATION)
    assert f"`LoDConfig.conf_low = {LoDConfig().conf_low}`" in text and f"**{EncoderConfig().det_thr}**" in text
    assert f"**{DEFAULT_RAW_CONF}**" in text
    assert LoDConfig().conf_high == PriorityConfig().p1_conf and f"**{LoDConfig().conf_high:.3f}**" in text
    for name in ("P7_DEMO_FORCE_FALLBACK", "P7_DEMO_ALLOW_THROTTLING", "demo/.streamlit/config.toml"):
        assert name in text


# ------------------------------------------------------------------------------ launcher, scripts, tiers
def test_launcher_starts_what_it_says(capsys):
    cmd, env = launch.plan("dashboard")
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("dashboard.py") and env == {}
    cmd, env = launch.plan("fallback", port=8600)
    assert env == {"P7_DEMO_FORCE_FALLBACK": "1"} and cmd[-2:] == ["--server.port", "8600"]
    assert launch.plan("quickstart")[0][-1].endswith("run_demo.py")
    assert launch.plan("summary")[0][-1].endswith("summary.py")
    assert launch.plan("gpu", full=True)[0][0] == "bash" and launch.plan("gpu", full=True)[0][-1] == "--full"
    for flag in ([], ["--fallback"], ["--quickstart"], ["--summary"], ["--gpu"]):
        assert launch.main(flag + ["--dry-run"]) == 0
    assert "streamlit run" in capsys.readouterr().out
    for mode, mods in launch.MODULES.items():
        if mode in launch.REQUIREMENTS:
            assert launch.REQUIREMENTS[mode].exists()
        assert set(launch.missing(mode)) <= set(mods)


def test_launcher_names_the_fix_when_a_package_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(launch, "missing", lambda mode: ["streamlit"])
    assert launch.main([]) == 2
    out = capsys.readouterr().out
    assert "no streamlit" in out and "pip install -r demo/requirements-dashboard.txt" in out and "--summary" in out


def test_launcher_check_reports_and_never_raises(capsys):
    assert launch.main(["--check"]) in (0, 1)
    out = capsys.readouterr().out
    assert "live dashboard:" in out and "static fallback:" in out and "power throttling:" in out


def test_simulations_script_runs_only_scripts_that_need_no_data():
    lines = (REPO / "code" / "rerun_simulations.sh").read_text(encoding="utf-8").splitlines()
    names = [re.match(r"run scripts/(\w+)\.py", ln).group(1) for ln in lines if ln.startswith("run scripts/")]
    assert len(names) == 19 and len(set(names)) == 16
    for needs_data in ("wp11_integration_demo", "wp18_campaign_runner", "wp24_detection_eval",
                       "wp25_semantic_packets", "wp26_b0_b4", "wp27_comms_direct_vs_relay"):
        assert needs_data not in names
    for name in set(names):
        src = (REPO / "code" / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import torch|from torch|from ultralytics|import onnxruntime)", src, re.M), name
    all_lines = (REPO / "code" / "rerun_all.sh").read_text(encoding="utf-8")
    for name in set(names) - {"contact_windows", "wp6_fit_size_model", "wp20_relay_path_choice"}:
        assert f"scripts/{name}.py" in all_lines, f"{name} is not in rerun_all.sh, the record of flags"
    assert "--measure" not in "".join(ln.split("#")[0] for ln in lines if ln.startswith("run "))


def test_the_three_test_tiers_are_registered_and_exclusive(request):
    ini = configparser.ConfigParser()
    ini.read(REPO / "code" / "pytest.ini", encoding="utf-8")
    markers = ini["pytest"]["markers"]
    for name in ("unit:", "inference:", "page:"):
        assert name in markers
    mine = request.node
    assert mine.get_closest_marker("unit") and not mine.get_closest_marker("inference")
    for item in request.session.items:
        tiers = [t for t in ("unit", "inference", "page") if item.get_closest_marker(t)]
        assert tiers and not ("unit" in tiers and len(tiers) > 1), (item.nodeid, tiers)
    items = request.session.items
    whole_suite = (not request.config.getoption("markexpr") and not request.config.getoption("keyword")
                   and len({i.path for i in items}) >= 25)
    if whole_suite:                                  # the counts the reproduction guide states
        unit = sum(bool(i.get_closest_marker("unit")) for i in items)
        text = _text(REPRODUCE)
        assert f"{len(items)} tests in `code/tests/`" in text, f"REPRODUCE.md: the suite has {len(items)} tests"
        assert f"| `unit` | {unit} |" in text and f"| `inference` | {len(items) - unit} |" in text
