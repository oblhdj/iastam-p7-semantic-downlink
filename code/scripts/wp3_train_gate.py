"""WP3 -- a tiny learned gate: "does this tile contain a ship?"

Why: on real Airbus tiles the classic computer-vision filter is not usable as a gate
(measured: recall 0.645 while dropping only 17.7% of empty tiles; dropping 42% costs
45% of the ships). The literature uses a small learned classifier instead, which gates
the expensive detector. This script trains one and measures it on the SAME axis, so the
two can be compared honestly.

    python scripts/wp3_train_gate.py --epochs 4 --size 128

Outputs: results/wp3_gate.json, results/wp3_gate_tradeoff.png|csv, models/gate.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def build_index(yolo_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    for f in sorted((yolo_dir / "images" / split).glob("*.jpg")):
        lbl = yolo_dir / "labels" / split / (f.stem + ".txt")
        n = 0
        sizes = []
        if lbl.exists():
            for line in lbl.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    n += 1
                    sizes.append(max(float(p[3]), float(p[4])) * 768)
        rows.append({"path": str(f), "n_ships": n, "max_ship_px": max(sizes) if sizes else 0.0})
    return pd.DataFrame(rows)



class Tiles(Dataset):
    """One tile -> (image tensor, has-ship label).

    Defined at module level on purpose: Windows starts DataLoader workers with `spawn`,
    which pickles the dataset, and a class declared inside main() cannot be pickled
    ("Can't get local object 'main.<locals>.Tiles'"). It worked on Linux/fork, so this
    only ever failed off Kaggle.
    """

    def __init__(self, df: "pd.DataFrame", train: bool, size: int):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1]
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        return x, torch.tensor(float(r.n_ships > 0))


class Gate(nn.Module):
    """~47k parameters: small enough for a satellite computer.

    Module level, like ``Tiles``: the integration demo (wp11) loads ``models/gate.pt`` and
    needs this class to rebuild the model. Nested inside main() it could not be imported,
    which meant every consumer copy-pasted it -- three divergent copies by 24 Sept.
    """

    def __init__(self, ch=(16, 32, 48, 64)):
        super().__init__()
        layers, prev = [], 3
        for c in ch:
            layers += [nn.Conv2d(prev, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(prev, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def load_gate(path, device="cpu"):
    """Rebuild the trained gate from a checkpoint. Returns (model, input_px)."""
    ck = torch.load(path, map_location="cpu")
    m = Gate()
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), int(ck["size"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--size", type=int, default=128, help="input resolution of the gate")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all tiles")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)


    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")


    tr = build_index(args.yolo_dir, "train")
    va = build_index(args.yolo_dir, "val")
    te = build_index(args.yolo_dir, "test")
    if args.limit_train:
        tr = tr.sample(min(args.limit_train, len(tr)), random_state=0)
    print(f"train {len(tr)} tiles ({tr.n_ships.gt(0).mean():.0%} with ships) | val {len(va)} | test {len(te)}")

    dl_tr = DataLoader(Tiles(tr, True, args.size), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_te = DataLoader(Tiles(te, False, args.size), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    model = Gate().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    pos_weight = torch.tensor([(tr.n_ships == 0).sum() / max(1, (tr.n_ships > 0).sum())], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    print(f"gate: {n_par/1000:.0f}k parameters, input {args.size}px")
    for ep in range(args.epochs):
        model.train()
        t0, tot, seen = time.perf_counter(), 0.0, 0
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(y)
            seen += len(y)
            if bi % 100 == 0:
                print(f"  epoch {ep+1} batch {bi}/{len(dl_tr)} loss {tot/max(1,seen):.4f}", flush=True)
        print(f"epoch {ep+1}/{args.epochs}: loss {tot/seen:.4f} in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- evaluate on test
    model.eval()
    scores, labels, ships = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in dl_te:
            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                s = torch.sigmoid(model(x.to(dev, non_blocking=True)))
            scores += s.float().cpu().tolist()
            labels += y.tolist()
    loop_ms = 1000 * (time.perf_counter() - t0) / len(te)
    # ^ that is the WHOLE eval loop: JPEG decode of 768x768 + resize + transfer + forward.
    #   It is dataloader-bound and ~80x the model's real cost, so it must NOT be compared
    #   against the detector's 10.7 ms/tile (pure GPU inference). Time the forward pass on a
    #   batch already resident on the device -- that is the number the break-even uses.
    def _pure_forward_ms(bs: int) -> float:
        x = torch.randn(bs, 3, args.size, args.size, device=dev)
        with torch.no_grad():
            for _ in range(10):
                model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(50):
                t1 = time.perf_counter()
                model(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t1) * 1000)
        return sorted(ts)[len(ts) // 2] / bs
    infer_ms = _pure_forward_ms(min(args.batch, 32))
    print(f"gate cost: {infer_ms:.3f} ms/tile pure forward on {dev} "
          f"({loop_ms:.2f} ms/tile for the dataloader-bound eval loop)")
    te = te.assign(score=scores)
    torch.save({"state_dict": model.state_dict(), "size": args.size}, ROOT / "models" / "gate.pt")

    # trade-off on the SAME axis as the classic filter:
    # x = share of empty tiles dropped, y = share of ships kept (on tiles we do not drop)
    total_ships = te.n_ships.sum()
    rows = []
    for thr in np.unique(np.round(np.linspace(0, 1, 201), 3)):
        keep = te.score >= thr
        ships_kept = te.loc[keep, "n_ships"].sum() / total_ships
        empties_dropped = (~keep & (te.n_ships == 0)).sum() / max(1, (te.n_ships == 0).sum())
        rows.append({"threshold": float(thr), "ship_recall": float(ships_kept),
                     "empty_dropped": float(empties_dropped), "tiles_kept": float(keep.mean())})
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(args.out / "wp3_gate_tradeoff.csv", index=False)

    def at_recall(r: float) -> dict:
        ok = tradeoff[tradeoff.ship_recall >= r]
        return ok.iloc[-1].to_dict() if len(ok) else {}

    p99, p995 = at_recall(0.99), at_recall(0.995)
    print(f"\ngate on TEST ({len(te)} tiles, {int(total_ships)} ships): {infer_ms:.2f} ms/tile on {dev}")
    print(f"  at 99.0% ship recall : drops {p99.get('empty_dropped', 0):.1%} of empty tiles "
          f"(threshold {p99.get('threshold', 0):.3f})")
    print(f"  at 99.5% ship recall : drops {p995.get('empty_dropped', 0):.1%} of empty tiles")
    print("  classic filter, same test data: recall 0.645 while dropping 17.7% of empty tiles")

    json.dump({"params": n_par, "input_px": args.size, "epochs": args.epochs,
               "infer_ms_per_tile": infer_ms, "eval_loop_ms_per_tile_dataloader_bound": loop_ms,
               "device": dev, "at_99_recall": p99, "at_99.5_recall": p995,
               "classic_baseline": {"ship_recall": 0.645, "empty_dropped": 0.177}},
              open(args.out / "wp3_gate.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(100 * tradeoff.empty_dropped, 100 * tradeoff.ship_recall, color="#1B6CA8", lw=2,
            label=f"learned gate ({n_par/1000:.0f}k parameters)")
    ax.scatter([17.7, 41.9, 58.1], [64.5, 55.5, 33.3], color="#C0392B", zorder=3,
               label="classic CV filter (measured)")
    ax.axhline(99, color="#2E8B57", ls=":", lw=1)
    ax.text(62, 99.6, "99% safety target", color="#2E8B57", fontsize=8)
    # NOT "compute saved": only 20.4% of tiles are empty, so dropping 40% of the empty ones
    # skips ~8% of detector calls. Labelling this axis "compute saved" overstates it ~5x.
    ax.set_xlabel("empty tiles dropped (%)   (20.4% of all tiles are empty, so 100% here = "
                  "20.4% of detector calls skipped)", fontsize=8)
    ax.set_ylabel("ships kept (%)  → safety")
    ax.set_title("Gate: how much compute can be saved without losing ships", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp3_gate_tradeoff.png", dpi=160)
    print(f"Saved wp3_gate.json, wp3_gate_tradeoff.csv|png in {args.out}")


if __name__ == "__main__":
    main()
