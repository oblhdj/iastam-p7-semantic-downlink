"""Write demo/fallback_assets/: what the dashboard shows when the live pipeline cannot run.

    python demo/make_fallback_assets.py            # regenerate from a live run on this machine
    python demo/make_fallback_assets.py --check    # verify the stored assets, change nothing

Runs the same demo_pipeline.run_all the dashboard runs live, at the default settings, on the
bundled synthetic swath and three representative tiles (ships, coast, cloud), and stores each
bundle with its visualisations. The assets carry no Airbus pixels (the samples are synthetic) and
no canonical number: those are always read from code/results. The manifest records the commit,
model fingerprint and the canon of the day, so a stale set is detected, not trusted.

Also the no-GPU smoke test of the whole chain: it prints the wall time, which must stay well
under a minute on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import demo_data as D  # noqa: E402
import host_power  # noqa: E402

TILE_SAMPLES = (("01_synthetic_00113a75c", "single tile — open sea, several ships"),
                ("04_synthetic_03204a586", "single tile — coast"),
                ("09_synthetic_024e6ba29", "single tile — cloud (everything dropped)"))
SWATH_TITLE = "3×3 synthetic swath, 2304 × 2304 px (shows SAHI against plain YOLO)"
MAX_VIS_SIDE = 1536                    # stored visualisations are downscaled: display, not data


def _shrink(jpg: bytes) -> bytes:
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    k = MAX_VIS_SIDE / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def generate(out: Path) -> int:
    env = D.environment()
    if not env["live_ok"]:
        print("The live pipeline cannot run here, so fallback assets cannot be generated:")
        for r in env["blocking"]:
            print(f"  missing: {r['name']} -- {r['detail']}\n    fix: {r['fix']}")
        return 2
    import demo_pipeline as P
    print("power throttling:", host_power.keep_full_speed())
    t_all = time.perf_counter()
    det, model = P.load_model()
    settings = P.Settings()
    samples = []
    tmp = out.with_name(out.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    jobs = [("swath", SWATH_TITLE)] + list(TILE_SAMPLES)
    for key, title in jobs:
        if key == "swath":
            img, info, _ = P.bundled_swath()
            note = ("9 bundled synthetic stand-in tiles (no Airbus pixels): the chain runs end to "
                    "end, the numbers are not measurements")
        else:
            name = key + ".jpg"
            img, info = P.read_image((D.QUICKSTART / "tiles" / name).read_bytes(), name)
            note = "a bundled synthetic stand-in tile (no Airbus pixels)"
        source = {"kind": "sample", "label": "SYNTH", "synthetic": True, "key": key, "note": note}
        bundle, art = P.run_all(img, info, det, settings, source=source, model_path=model)
        if bundle["comms"].get("error"):
            print(f"link simulation failed for {key}: {bundle['comms']['error']}")
            return 2
        d = tmp / key
        d.mkdir(parents=True)
        (d / "bundle.json").write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
        for fname, data in art.items():
            (d / fname).write_bytes(_shrink(data) if fname.endswith(".jpg") else data)
        m = bundle["modes"]["B3"]
        samples.append({"key": key, "title": title, "files": sorted(art),
                        "B3_bytes": m["bytes"]["total"], "B3_levels": m["level_histogram"],
                        "elapsed_s": bundle["meta"]["elapsed_s"]})
        print(f"  {key:<26} {info.width}x{info.height}  B3 packet {m['bytes']['total']:>7,} B  "
              f"{bundle['meta']['elapsed_s']:.2f} s")
    manifest = {"bundle_version": D.BUNDLE_VERSION,
                "what": "pre-generated dashboard results from a known-good live run; shown when the "
                        "live pipeline cannot run. SYNTH: synthetic stand-in images, not measurements.",
                "generated_at": bundle["meta"]["generated_at"], "commit": bundle["meta"]["commit"],
                "backend": bundle["meta"]["backend"], "model_sha256_16": bundle["meta"]["model_sha256_16"],
                "settings": bundle["settings"], "canon_at_generation": D.canon_snapshot(),
                "samples": samples}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out.relative_to(D.REPO)} ({size / 1e6:.2f} MB, {len(samples)} samples) in "
          f"{time.perf_counter() - t_all:.1f} s, model load included")
    return check(out)


def check(out: Path) -> int:
    """Every stored sample loads, is self-consistent, and was generated beside today's canon."""
    index = D.fallback_index(out)
    if "error" in index:
        print("FAIL:", index["error"])
        return 1
    bad = []
    for s in index["samples"]:
        try:
            bundle, files = D.load_fallback(s["key"], out)
        except (FileNotFoundError, ValueError) as e:
            bad.append(f"{s['key']}: {e}")
            continue
        for mode in D.MODES:
            b = bundle["modes"][mode]["bytes"]
            if b["payload"] + b["overhead"] != b["total"]:
                bad.append(f"{s['key']} {mode}: payload + overhead != total")
            for route in ("direct", "policy", "relay"):
                D.mode_summary(bundle, mode, route)
        if len(files["downlink_B3.bin"]) != bundle["modes"]["B3"]["bytes"]["total"]:
            bad.append(f"{s['key']}: downlink_B3.bin is not the size the bundle states")
    stale = D.fallback_staleness(index)
    for line in bad:
        print("FAIL:", line)
    for line in stale:
        print("STALE (canon moved since generation):", line)
    if not bad and not stale:
        print(f"fallback assets OK: {len(index['samples'])} samples, generated {index['generated_at']} "
              f"at {index['commit']}, canon unchanged since")
    return 1 if bad or stale else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the stored assets, change nothing")
    ap.add_argument("--out", type=Path, default=D.FALLBACK)
    a = ap.parse_args()
    sys.exit(check(a.out) if a.check else generate(a.out))
