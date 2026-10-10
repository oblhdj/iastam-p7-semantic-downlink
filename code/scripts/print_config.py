"""Print every configuration object of the pipeline with its default value, as Markdown.

    python scripts/print_config.py                                  # print the tables
    python scripts/print_config.py --write ../docs/CONFIGURATION.md  # refresh the block in the document
    python scripts/print_config.py --check ../docs/CONFIGURATION.md  # exit 1 if that block is out of date

Why it exists: docs/CONFIGURATION.md must not drift from the code. The defaults below are read from
the dataclasses themselves and the "meaning" column is the comment written next to each field in
the source, so the document's table is regenerated, never retyped; tests/test_docs.py runs --check.
Needs nothing beyond the analysis environment (no torch, no dataset, no weights).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "demo"))

BEGIN = "<!-- BEGIN GENERATED: python code/scripts/print_config.py -->"
END = "<!-- END GENERATED -->"

# (heading, module, class, one line on where it is used)
SECTIONS = [
    ("Perception", "sat7.perception", "PerceptionConfig", "how the detector sees one frame: B1 (`whole`) or B2 (`sahi`)"),
    ("Perception", "sat7.b2_sahi_fusion", "SahiConfig", "the window grid and fusion rule `PerceptionConfig` drives"),
    ("Pre-filter", "sat7.prefilter", "PrefilterConfig", "the classic CV stage that supplies each tile's context (cloud / coast / sea)"),
    ("Semantic packet", "sat7.semantic", "EncoderConfig", "what is sent for a detection and how large it is (records, ROI and context crops, packets)"),
    ("Semantic packet", "sat7.priority", "PriorityConfig", "the P0-P3 policy of the paper's Table I"),
    ("Semantic packet", "sat7.semantic", "TMFraming", "CCSDS TM link layer, used only to report on-air bytes"),
    ("Level of detail and scheduling", "sat7.scheduler", "LoDConfig", "the level-of-detail ladder behind the headline (sizes, thresholds, values)"),
    ("Level of detail and scheduling", "sat7.scheduler", "SizeModel", "the measured power law for crop sizes (fitted by wp6_fit_size_model.py)"),
    ("Level of detail and scheduling", "sat7.real_workload", "RealWorkloadConfig", "the simulated day built from the detection catalogue"),
    ("Orbit and links", "sat7.orbit", "OrbitConfig", "the primary satellite's orbit"),
    ("Orbit and links", "sat7.orbit", "GroundStation", "the ground station"),
    ("Orbit and links", "sat7.orbit", "LinkConfig", "the ground-link budget model (preset `cubesat_sband`)"),
    ("Orbit and links", "sat7.comms", "CommsConfig", "the two-path (direct / relay) link simulator: shares, rates, powers, routing"),
    ("Orbit and links", "sat7.relay", "RelayConfig", "the earlier window-only relay model (the canonical B4 row)"),
    ("Joint program", "sat7.joint", "JointWeights", "alpha / beta / gamma of the paper's eq 20"),
    ("Joint program", "sat7.joint", "JointConstraints", "the two floors of eq 20"),
    ("Joint program", "sat7.joint", "CostModel", "bytes to energy and latency inside the joint program"),
    ("Dashboard", "demo_pipeline", "Settings", "what the dashboard's sidebar sets for one run"),
]
FIELD = re.compile(r"^(\s+)(\w+)\s*:\s*[^=#]+?(?:=\s*.+?)?\s*(?:#\s*(.*))?$")


def _comments(cls) -> dict[str, str]:
    """The comment written beside each field in the source (continuation lines included)."""
    out, current, indent = {}, None, 0
    for line in inspect.getsource(cls).splitlines():
        m = FIELD.match(line)
        if m and not line.lstrip().startswith(("#", "def ", "class ", '"""', "@")):
            current, indent = m.group(2), len(m.group(1))
            out[current] = (m.group(3) or "").strip()
            continue
        s = line.strip()
        lead = len(line) - len(line.lstrip())
        if current and s.startswith("#") and lead > indent + 4:       # aligned under the comment
            out[current] = (out[current] + " " + s.lstrip("# ").strip()).strip()
        elif s and not s.startswith("#"):
            current = None
    return out


def _show(value) -> str:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return f"{type(value).__name__}(...)"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, float):
        return f"{value:g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.6g}"
    return repr(value)


def _defaults(cls):
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        if f.default is not dataclasses.MISSING:
            yield f.name, _show(f.default)
        elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
            yield f.name, _show(f.default_factory())                # type: ignore[misc]
        else:
            yield f.name, "(required)"


def render() -> str:
    lines, heading = [], None
    for head, module, name, used in SECTIONS:
        try:
            cls = getattr(importlib.import_module(module), name)
        except Exception as e:  # noqa: BLE001  the dashboard module needs OpenCV; say so, keep going
            lines += [f"#### `{module}.{name}`", "", f"_Not importable here ({type(e).__name__}: {e})._", ""]
            continue
        if head != heading:
            lines += [f"### {head}", ""]
            heading = head
        notes = _comments(cls)
        lines += [f"#### `{module}.{name}`", "", f"{used[0].upper()}{used[1:]}.", "",
                  "| field | default | meaning (the comment in the source) |", "|---|---|---|"]
        for field, default in _defaults(cls):
            note = notes.get(field, "").replace("|", "\\|")
            lines.append(f"| `{field}` | `{default}` | {note} |")
        lines.append("")
    from sat7 import energy, orbit, scheduler
    lines += ["### Constants that are not dataclass fields", "",
              "| name | value | meaning |", "|---|---|---|"]
    for k, v in energy.DEFAULT_POWERS_W.items():
        lines.append(f"| `sat7.energy.DEFAULT_POWERS_W[{k!r}]` | `{v:g}` W | ASSUMPTION: no power was measured; swept in wp17 |")
    lines.append(f"| `sat7.energy.DEFAULT_R_TX_BPS` | `{energy.DEFAULT_R_TX_BPS:g}` bit/s | SIM: effective downlink rate from the orbit simulation |")
    for name, link in orbit.LINK_PRESETS.items():
        lines.append(f"| `sat7.orbit.LINK_PRESETS[{name!r}]` | {link.bandwidth_hz / 1e6:g} MHz, SNR "
                     f"{link.snr_zenith_db:g} dB at zenith, cap {link.max_rate_bps / 1e6:g} Mbps, efficiency "
                     f"{link.efficiency:g} | link-budget preset" + (" (the default)" if name == "cubesat_sband" else "") + " |")
    lines.append(f"| `sat7.scheduler.RAW_TILE_BYTES` | `{scheduler.RAW_TILE_BYTES:,}` B | one 768 x 768 tile as uncompressed 8-bit RGB: the raw baseline |")
    lines.append(f"| `sat7.scheduler.COAST_TILE_BYTES_MEASURED` | `{scheduler.COAST_TILE_BYTES_MEASURED:,.0f}` B | mean measured size of a coastal context tile (wp28) |")
    lines.append("")
    return "\n".join(lines)


def block() -> str:
    return f"{BEGIN}\n\n{render()}\n{END}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", type=Path, help="replace the generated block in this Markdown file")
    ap.add_argument("--check", type=Path, help="exit 1 if this file's generated block differs")
    a = ap.parse_args()
    target = a.write or a.check
    if target is None:
        sys.stdout.reconfigure(encoding="utf-8")
        print(render())
        return 0
    text = target.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{target}: no generated block ({BEGIN} ... {END})")
        return 1
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    fresh = head + block() + tail
    if a.check:
        ok = fresh == text
        print(f"{target.name}: generated configuration block " + ("is up to date" if ok else "is OUT OF DATE; rerun with --write"))
        return 0 if ok else 1
    target.write_text(fresh, encoding="utf-8", newline="\n")
    print(f"{target.name}: generated block rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
