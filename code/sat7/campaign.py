"""B0-B4 as selectable modes of ONE onboard pipeline over ONE shared input set (paper Table II).

The paper's five configurations are not five programs; they are one chain with three switches:

    perception  none | whole | sahi                 what the detector sees
    payload     raw_image | detections | semantic   what goes on the downlink
    relay       False | True                        may the optional inter-satellite relay carry it

    mode  perception  payload     relay   paper definition
    B0    none        raw_image   no      no onboard detection, transmit the full image
    B1    whole       detections  no      YOLO on the original image, transmit detection output
    B2    sahi        detections  no      SAHI + YOLO, transmit detection output
    B3    sahi        semantic    no      SAHI + YOLO + semantic policy, metadata + optional ROI
    B4    sahi        semantic    yes     B3 + direct / optional relay communication

Every stage is an existing sat7 component -- perception.detect_image (whole / SAHI + fusion),
semantic.encode_image (records, ROI / context crops, CCSDS packets), semantic.encode_raw_image
(B0), scheduler.simulate, comms.simulate_comms (B4's two-path link simulator, SIM) -- so a mode
differs from another ONLY in its switches and
every mode sees the same scene pixels, detector, onboard cut, matching rule and packetization.

  * "detection output" (B1, B2) is the semantic record with no policy: every detection at or above
    the onboard cut becomes one P1 record (DETECTIONS_ONLY), no ROI, no context, no AIS bump.
  * B1 runs ONE detector call on the original image (the letterbox scales a 3072 px scene by 1/4);
    there is no slicing stage. Running one call per 768 px source tile -- what wp18 called B1 -- is a
    16-window non-overlapping tiling of the scene, i.e. a slicing stage; it is kept only as a
    labelled reference row ("B1_per_tile"), not as B1.
  * B3 assigns each fused SAHI box to the captured tile its centre falls in, then applies the
    P0-P3 policy with that tile's pre-filter context (cloud tiles discard, coastal tiles escalate).
    ROI crops are cut from the scene, so a ship on a tile seam is not clipped; a coastal context
    image is the source tile (context_box).

Scenes are N x N real test tiles stitched abutting (the wp15 convention), with tile contexts drawn
by the orbit mix the day simulations assume (15% cloud / 20% ships / 5% coast / 60% empty sea).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .perception import PerceptionConfig, detect_image
from .priority import PriorityConfig
from .semantic import EncoderConfig, Ids, Packetizer, encode_image, encode_raw_image

TILE = 768
CONTEXTS = ("cloud", "ships", "coast", "empty")
ORBIT_MIX = (0.15, 0.20, 0.05, 0.60)          # RealWorkloadConfig.orbit_mix (ASSUMPTION, swept in report 11)
ONBOARD_CTX = {"cloud": "cloud", "land_coast": "coast"}      # pre-filter verdict -> policy context

# every detection at or above the cut is a P1 record: "detection output", no semantic policy
DETECTIONS_ONLY = PriorityConfig(p1_conf=0.0, coast_to_p3=False, dark_bump=False)


@dataclass(frozen=True)
class BMode:
    name: str
    perception: str        # none | whole | sahi
    payload: str           # raw_image | detections | semantic
    relay: bool
    paper: str


MODES = {
    "B0": BMode("B0", "none", "raw_image", False, "no onboard detection, transmit the full image"),
    "B1": BMode("B1", "whole", "detections", False, "YOLO on the original image, transmit detection output"),
    "B2": BMode("B2", "sahi", "detections", False, "SAHI + YOLO, transmit detection output"),
    "B3": BMode("B3", "sahi", "semantic", False, "SAHI + YOLO + semantic policy, metadata + optional ROI"),
    "B4": BMode("B4", "sahi", "semantic", True, "B3 + direct / optional relay communication"),
}


@dataclass
class SceneTile:
    row: int
    col: int
    image: str             # the real test tile
    context: str           # onboard (pre-filter) context: cloud | coast | ships
    drawn_as: str          # the catalogue context it was drawn from (cloud | ships | coast | empty)


@dataclass
class Scene:
    scene_id: int
    per_side: int
    tiles: list[SceneTile]
    gts: list[tuple] = field(default_factory=list)      # ground truth, scene pixels (cx, cy, w, h)
    gt_tile: list[int] = field(default_factory=list)    # index into tiles

    @property
    def size(self) -> int:
        return self.per_side * TILE

    def tile_box(self, k: int) -> tuple[int, int, int, int]:
        t = self.tiles[k]
        return (t.col * TILE, t.row * TILE, (t.col + 1) * TILE, (t.row + 1) * TILE)

    def tile_index(self, cx: float, cy: float) -> int:
        c = min(self.per_side - 1, max(0, int(cx // TILE)))
        r = min(self.per_side - 1, max(0, int(cy // TILE)))
        return r * self.per_side + c


def sample_scenes(pools: dict, onboard_ctx: dict, gts_of, n: int, per_side: int = 4,
                  mix=ORBIT_MIX, seed: int = 0) -> list[Scene]:
    """n scenes of per_side x per_side real tiles, contexts drawn by `mix` (empty pools dropped).

    `pools[ctx]` lists the real tiles of each catalogue context; `gts_of(name)` returns a tile's
    ground truth in tile pixels; `onboard_ctx[name]` is its pre-filter context."""
    rng = np.random.default_rng(seed)
    p = np.array([mix[i] if pools.get(c) else 0.0 for i, c in enumerate(CONTEXTS)], float)
    p /= p.sum()
    scenes = []
    for sid in range(n):
        drawn = rng.choice(CONTEXTS, size=per_side * per_side, p=p)
        tiles, gts, gt_tile = [], [], []
        for k, ctx in enumerate(drawn):
            name = str(pools[ctx][rng.integers(len(pools[ctx]))])
            r, c = divmod(k, per_side)
            tiles.append(SceneTile(r, c, name, onboard_ctx[name], str(ctx)))
            for (x, y, w, h) in gts_of(name):
                gts.append((x + c * TILE, y + r * TILE, w, h))
                gt_tile.append(k)
        scenes.append(Scene(sid, per_side, tiles, gts, gt_tile))
    return scenes


def render(scene: Scene, load_tile) -> np.ndarray:
    """The scene's pixels, stitched from its tiles (abutting, as wp15). `load_tile(name)` -> BGR."""
    img = np.zeros((scene.size, scene.size, 3), np.uint8)
    for t in scene.tiles:
        img[t.row * TILE:(t.row + 1) * TILE, t.col * TILE:(t.col + 1) * TILE] = load_tile(t.image)
    return img


def perceive(img, mode: BMode, detect_fn, window: int = TILE, overlap: float = 0.20) -> list:
    """The perception switch: no detector (B0), one call on the original image (B1), SAHI (B2-B4)."""
    if mode.perception == "none":
        return []
    if mode.perception == "whole":
        return detect_image(img, detect_fn, PerceptionConfig(mode="whole"))
    return detect_image(img, detect_fn, PerceptionConfig(mode="sahi", window=window, overlap=overlap))


@dataclass
class Downlink:
    products: list            # semantic.Product
    det_index: list           # per product: global detection indices it carries
    levels: list | None = None   # per global detection (semantic payload only), P0..P3


def downlink(scene: Scene, img, dets: list, mode: BMode, cfg: EncoderConfig | None = None,
             t_capture_s: float = 0.0, dark=None, jpeg_fn=None, packetizer: Packetizer | None = None,
             ids: Ids | None = None) -> Downlink:
    """The payload switch. `img` may be a shape-only array when `jpeg_fn` serves every crop (B1-B4);
    B0 needs the real pixels."""
    cfg = cfg or EncoderConfig()
    pk, ids = packetizer or Packetizer(cfg.max_packet_data), ids or Ids()
    if mode.payload == "raw_image":
        prod = encode_raw_image(img, scene.scene_id, pk)
        return Downlink([prod], [()])
    if mode.payload == "detections":
        enc = encode_image(img, scene.scene_id, t_capture_s, dets, "ships",
                           replace(cfg, priority=DETECTIONS_ONLY), ids, pk, jpeg_fn=jpeg_fn)
        return Downlink(enc.products, [p.dets for p in enc.products], enc.levels)
    # semantic: per captured tile, with that tile's context
    dark = list(dark) if dark is not None else [False] * len(dets)
    by_tile: dict[int, list[int]] = {}
    for i, d in enumerate(dets):
        by_tile.setdefault(scene.tile_index(d[0], d[1]), []).append(i)
    products, det_index, levels = [], [], [0] * len(dets)
    for k, members in sorted(by_tile.items()):
        t = scene.tiles[k]
        enc = encode_image(img, scene.scene_id * scene.per_side ** 2 + k, t_capture_s,
                           [dets[i] for i in members], t.context, cfg, ids, pk,
                           dark=[dark[i] for i in members], jpeg_fn=jpeg_fn,
                           context_box=scene.tile_box(k))
        for local, lv in enumerate(enc.levels):
            levels[members[local]] = lv
        for p in enc.products:
            products.append(p)
            det_index.append(tuple(members[j] for j in p.dets))
    return Downlink(products, det_index, levels)
