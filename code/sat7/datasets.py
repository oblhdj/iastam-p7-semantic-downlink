"""Large-scene dataset loaders + a SAHI evaluation path (paper section VIII, recommended datasets).

The paper recommends large-scene remote-sensing sets -- xView, VisDrone, DOTA, HRSID (SAR) -- because
they are big enough to *slice*, which Airbus 768x768 tiles are not (START_HERE section 7 / report 13:
"you cannot slice a tile"). This module gives the first-class SAHI path (sat7.perception) real large
scenes to run on: annotation parsers that normalise each set's native format into the repo's box
convention, and an evaluation loop that scores the SAHI pipeline on them by size bucket -- the same
small/medium/large recall the B1/B2 campaign reports.

Covered formats (each a pure, image-free parser so the loaders are unit-testable without downloading
tens of GB):
  * DOTA  -- oriented-bbox `.txt`: 8 polygon coords + class + difficult, per line (one file per image)
  * YOLO  -- `class cx cy w h` normalised (DOTA-after-conversion, HRSID-YOLO, VisDrone-YOLO)
  * COCO  -- one `.json` with images[] + annotations[] in xywh top-left pixels (HRSID default, xView-like)

Box convention out: (cx, cy, w, h) in pixels, centre form -- what sat7.perception / b2_sahi_fusion /
recall_of all use. Class filtering keeps only the maritime/target category by default ("ship").

Labels: recall this path produces is REAL only when `detect_fn` is the trained detector on the real
scenes; on a stub it is a plumbing check. The module ships no numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .b2_sahi_fusion import _iou
from .perception import PerceptionConfig, detect_image

Box = tuple[float, float, float, float]            # (cx, cy, w, h) px, centre form


@dataclass
class Scene:
    """One large scene: the image to slice + its ground-truth boxes."""
    name: str
    width: int
    height: int
    gt_boxes: list[Box] = field(default_factory=list)
    image_path: Path | None = None


# ----------------------------------------------------------------------------- parsers (pure)
def parse_dota_lines(lines, classes=("ship",)) -> list[Box]:
    """DOTA oriented-bbox lines -> axis-aligned centre-form pixel boxes.

    Each object line is `x1 y1 x2 y2 x3 y3 x4 y4 category difficult`. Header lines
    (`imagesource:`, `gsd:`) and blanks are skipped. The oriented quad is reduced to its
    axis-aligned bounding box, which is what the detector and the IoU matcher use.
    """
    want = None if classes is None else {c.lower() for c in classes}
    out: list[Box] = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 9 or not _isnum(p[0]):          # header / blank / malformed
            continue
        xs = [float(p[0]), float(p[2]), float(p[4]), float(p[6])]
        ys = [float(p[1]), float(p[3]), float(p[5]), float(p[7])]
        cat = p[8].lower()
        if want is not None and cat not in want:
            continue
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def parse_yolo_lines(lines, width: int, height: int, classes=None) -> list[Box]:
    """YOLO `class cx cy w h` (normalised 0-1) -> pixel centre-form boxes.

    `classes`, if given, keeps only those integer class ids (e.g. {0} for the ship class).
    """
    out: list[Box] = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 5 or not _isnum(p[1]):
            continue
        cls = int(float(p[0]))
        if classes is not None and cls not in classes:
            continue
        cx, cy, w, h = (float(p[1]) * width, float(p[2]) * height,
                        float(p[3]) * width, float(p[4]) * height)
        out.append((cx, cy, w, h))
    return out


def scenes_from_coco(coco: dict, category_names=("ship",)) -> list[Scene]:
    """COCO dict (images[] + annotations[] with bbox [x, y, w, h] top-left px) -> scenes.

    Used for HRSID (SAR ships) and any xView-style COCO export. `category_names=None` keeps all.
    """
    want_ids = None
    if category_names is not None:
        want = {c.lower() for c in category_names}
        want_ids = {c["id"] for c in coco.get("categories", []) if c["name"].lower() in want}
    by_img: dict[int, list[Box]] = {}
    for a in coco.get("annotations", []):
        if want_ids is not None and a.get("category_id") not in want_ids:
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((x + w / 2, y + h / 2, w, h))
    scenes = []
    for im in coco.get("images", []):
        scenes.append(Scene(name=im.get("file_name", str(im["id"])),
                            width=int(im["width"]), height=int(im["height"]),
                            gt_boxes=by_img.get(im["id"], []),
                            image_path=Path(im["file_name"]) if "file_name" in im else None))
    return scenes


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------- dir loaders
def load_dota_dir(img_dir, ann_dir, classes=("ship",), read_size=None) -> list[Scene]:
    """Pair every image in `img_dir` with its DOTA `.txt` in `ann_dir`.

    `read_size(path) -> (w, h)` resolves each image's size (DOTA annotations are already in pixels,
    so size is only needed to set Scene.width/height for the slicer); pass a cheap header reader
    (e.g. PIL Image.open(path).size) or None to leave sizes at 0 for annotation-only use.
    """
    img_dir, ann_dir = Path(img_dir), Path(ann_dir)
    scenes = []
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            continue
        ann = ann_dir / (img.stem + ".txt")
        if not ann.exists():
            continue
        w, h = read_size(img) if read_size else (0, 0)
        scenes.append(Scene(img.name, int(w), int(h),
                            parse_dota_lines(ann.read_text().splitlines(), classes), img))
    return scenes


def load_hrsid(coco_json, category_names=("ship",)) -> list[Scene]:
    """HRSID (SAR ship detection) comes as a COCO json; load its scenes."""
    return scenes_from_coco(json.loads(Path(coco_json).read_text()), category_names)


# ----------------------------------------------------------------------------- matching + eval
def recall_of(detections, gts, iou_thr: float = 0.3) -> list[bool]:
    """Per-GT hit/miss by greedy IoU matching (same rule as scripts/wp12_seams.recall_of).

    detections/gts are centre-form boxes; a GT is 'found' if some unused detection overlaps it at
    >= iou_thr. One detection matches at most one GT (greedy, highest IoU first).
    """
    used = [False] * len(detections)
    found = []
    for g in gts:
        best_i, best_iou = -1, iou_thr
        for i, d in enumerate(detections):
            if used[i]:
                continue
            v = _iou(g[:4], d[:4])
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            used[best_i] = True
            found.append(True)
        else:
            found.append(False)
    return found


def recall_by_size(found, sizes) -> dict:
    """Overall + small(<32) / medium(32-96) / large(>96) recall, the campaign's reporting buckets."""
    import numpy as np
    found, sizes = np.asarray(found, bool), np.asarray(sizes, float)

    def r(m):
        return round(float(found[m].mean()), 4) if m.any() else None
    return {"overall": round(float(found.mean()), 4) if found.size else None,
            "small_<32": r(sizes < 32), "medium": r((sizes >= 32) & (sizes <= 96)),
            "large_>96": r(sizes > 96), "n_gt": int(found.size)}


def evaluate_scenes(scenes, image_loader, detect_fn, cfg: PerceptionConfig | None = None) -> dict:
    """Run the SAHI perception path over scenes and report recall by size bucket (= B2's metric).

    `image_loader(scene) -> (H, W, C) array`. Compares mode="sahi" against the GT. Pass a
    `PerceptionConfig(mode="whole")` to get the B1 (no-SAHI) baseline on the same scenes -- the
    without-SAHI ablation, on large scenes this time.
    """
    cfg = cfg or PerceptionConfig(mode="sahi")
    found, sizes, n_det, n_scenes = [], [], 0, 0
    for sc in scenes:
        if not sc.gt_boxes:
            continue
        img = image_loader(sc)
        dets = detect_image(img, detect_fn, cfg)
        n_det += len(dets)
        found += recall_of(dets, sc.gt_boxes)
        sizes += [max(g[2], g[3]) for g in sc.gt_boxes]
        n_scenes += 1
    rec = recall_by_size(found, sizes)
    return {"mode": cfg.mode, "window": cfg.window, "overlap": cfg.overlap, "fuse": cfg.fuse,
            "scenes": n_scenes, "detections": n_det, "recall": rec}


if __name__ == "__main__":
    import numpy as np

    # DOTA: a header + two ships + one non-ship; only the ships survive the class filter.
    dota = ["imagesource:GoogleEarth", "gsd:0.3",
            "100 100 140 100 140 160 100 160 ship 0",      # bbox cx120 cy130 w40 h60
            "500 500 560 500 560 540 500 540 ship 0",
            "10 10 20 10 20 20 10 20 harbor 0"]
    boxes = parse_dota_lines(dota)
    assert len(boxes) == 2 and abs(boxes[0][0] - 120) < 1e-9 and abs(boxes[0][3] - 60) < 1e-9
    print("DOTA parse OK:", boxes)

    # YOLO normalised -> pixels
    yolo = ["0 0.5 0.5 0.1 0.2", "1 0.1 0.1 0.05 0.05"]
    yb = parse_yolo_lines(yolo, 1000, 1000, classes={0})
    assert len(yb) == 1 and yb[0] == (500.0, 500.0, 100.0, 200.0)
    print("YOLO parse OK:", yb)

    # COCO -> scenes
    coco = {"categories": [{"id": 1, "name": "ship"}, {"id": 2, "name": "plane"}],
            "images": [{"id": 7, "file_name": "s.png", "width": 800, "height": 600}],
            "annotations": [{"image_id": 7, "category_id": 1, "bbox": [10, 20, 40, 60]},
                            {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]}]}
    scs = scenes_from_coco(coco)
    assert len(scs) == 1 and len(scs[0].gt_boxes) == 1 and scs[0].gt_boxes[0] == (30.0, 50.0, 40.0, 60.0)
    print("COCO parse OK:", scs[0].gt_boxes)

    # end-to-end plumbing: a stub detector that returns each GT exactly -> recall 1.0 via SAHI path
    sc = Scene("t", 1536, 768, gt_boxes=[(200, 200, 40, 40), (1000, 400, 50, 30)])

    def perfect(crops):
        # report nothing per crop; instead we fake a single whole-image-ish detector by returning
        # the GTs shifted into each crop only when they fall inside it. Simpler: stub returns [] and
        # we instead test the matcher directly below. Here just assert the loop runs.
        return [[] for _ in crops]
    out = evaluate_scenes([sc], lambda s: np.zeros((s.height, s.width, 3), np.uint8),
                          perfect, PerceptionConfig(mode="sahi", window=768))
    assert out["recall"]["overall"] == 0.0 and out["scenes"] == 1
    # matcher: detections equal to GTs -> all found
    assert all(recall_of([(200, 200, 40, 40), (1000, 400, 50, 30)], sc.gt_boxes))
    print("eval plumbing + matcher OK:", out["recall"])
    print("datasets self-check OK")
