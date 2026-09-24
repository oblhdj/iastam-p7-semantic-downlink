"""Score every catalogue tile with the trained gate, so downstream work can use its signal.

WP7's thumbnail gating and WP8's control law both key on the *classic* pre-filter: "empty_sea
and the detector fired nothing" for gating, and "bright objects the classic stage saw that the
network did not confirm" for coastal escalation. That classic signal was measured at 0.645
ship recall (WP3) and costs 51 ms/tile (WP11) -- the learned gate reaches 0.988 at 0.05 ms.

This writes one score per tile so both can be re-keyed on the better signal.

    .venv312/Scripts/python.exe scripts/wp3_score_tiles.py --split test

Outputs: results/wp3_gate_scores_<split>.csv  (image, gate_score)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np
import pandas as pd
import torch

from wp3_train_gate import load_gate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--gate", type=Path, default=ROOT / "models" / "gate.pt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, px = load_gate(args.gate, dev)
    files = sorted((args.data / "images" / args.split).glob("*.jpg"))
    print(f"scoring {len(files)} tiles of split '{args.split}' on {dev}, input {px}px")

    names, scores = [], []
    t0 = time.perf_counter()
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        arr = np.stack([cv2.resize(cv2.imread(str(f), cv2.IMREAD_COLOR), (px, px),
                                   interpolation=cv2.INTER_AREA) for f in chunk])
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float().div_(255.0).to(dev)
        with torch.no_grad():
            s = torch.sigmoid(model(x)).float().cpu().numpy()
        names += [f.name for f in chunk]
        scores += s.tolist()
        if i % (args.batch * 10) == 0:
            print(f"  {i + len(chunk)}/{len(files)}", flush=True)

    df = pd.DataFrame({"image": names, "gate_score": scores})
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"wp3_gate_scores_{args.split}.csv"
    df.to_csv(path, index=False)
    dt = time.perf_counter() - t0
    print(f"\n{len(df)} tiles in {dt:.0f}s ({1000*dt/len(df):.1f} ms/tile incl. JPEG decode)")
    print(f"score distribution: min {df.gate_score.min():.4f} median "
          f"{df.gate_score.median():.4f} max {df.gate_score.max():.4f}")
    print(f"Saved {path.name} in {args.out}")


if __name__ == "__main__":
    main()
