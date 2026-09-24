"""Near-duplicate detection for Airbus tiles (WP0).

Why: Airbus 768x768 tiles are crops of larger scenes and neighbouring crops overlap,
so the same ship can appear in several images. A random train/test split then leaks
answers into the test set and inflates every score we report.

Pipeline
--------
1. **Signatures**  perceptual hash (pHash) of the whole tile *and* of overlapping
   windows, plus a small thumbnail vector.
2. **Candidate search**  nearest neighbours on the thumbnail vectors (cosine, chunked
   matrix product) *and* LSH banding on the hashes. Banding alone is not enough: it
   needs an exact 16-bit match, which a shifted crop never produces -- the very case
   we are hunting. The thumbnail search degrades smoothly with a shift.
3. **Proof of overlap (mandatory)**  hashes are NOT evidence: on near-featureless open
   sea two unrelated tiles easily land within a few bits of each other (we saw exactly
   this: different ships, different water, Hamming distance 2). So every candidate is
   verified: phase correlation estimates the shift between the two tiles, then the
   overlapping region must actually correlate (normalised cross-correlation).
4. **Second opinion**  borderline pairs go to ORB keypoints + RANSAC homography.
5. **Grouping**  union-find: a ~ b and b ~ c put a, b, c in one group. A group is
   never split across train/val/test.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- signatures

WINDOWS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 1.00, 1.00),    # whole tile
    (0.00, 0.00, 0.60, 0.60),    # 4 overlapping corners: a shifted crop keeps one of them
    (0.40, 0.00, 1.00, 0.60),
    (0.00, 0.40, 0.60, 1.00),
    (0.40, 0.40, 1.00, 1.00),
    (0.20, 0.20, 0.80, 0.80),    # centre
)


def phash(gray: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.uint64:
    """64-bit perceptual hash: DCT of a 32x32 thumbnail, low frequencies vs their median."""
    side = hash_size * highfreq
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:hash_size, :hash_size].flatten()
    bits = low > np.median(low[1:])          # skip the DC term (overall brightness)
    return np.packbits(bits).view(">u8")[0]


def thumb_vector(gray: np.ndarray, side: int = 8) -> np.ndarray:
    """Contrast-normalised thumbnail, flattened: the feature used to find candidates.

    Two tiles of the same scene shifted by a few percent keep very similar thumbnails,
    while unrelated sea tiles do not.
    """
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    n = np.linalg.norm(small)
    return small / n if n > 1e-6 else small


def knn_pairs(vectors: np.ndarray, k: int = 12, chunk: int = 512) -> set[tuple[int, int]]:
    """Candidate pairs: the k most similar thumbnails of each image (cosine similarity)."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    n = len(x)
    if n < 2:
        return set()
    k = min(k, n - 1)
    pairs: set[tuple[int, int]] = set()
    for start in range(0, n, chunk):
        sims = x[start:start + chunk] @ x.T
        for row_i, row in enumerate(sims):
            i = start + row_i
            row[i] = -np.inf
            for j in np.argpartition(-row, k - 1)[:k]:
                j = int(j)
                if row[j] > -np.inf:
                    pairs.add((i, j) if i < j else (j, i))
    return pairs


def signatures(gray: np.ndarray) -> list[np.uint64]:
    """One pHash per window (see WINDOWS)."""
    h, w = gray.shape[:2]
    out = []
    for x0, y0, x1, y1 in WINDOWS:
        crop = gray[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if crop.size:
            out.append(phash(crop))
    return out


def thumb_gray(gray: np.ndarray, side: int = 128) -> np.ndarray:
    """Small grey copy kept in memory for the overlap proof (no re-decoding later)."""
    return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)


_HANN: dict[tuple[int, int], np.ndarray] = {}


def _hann(h: int, w: int) -> np.ndarray:
    if (h, w) not in _HANN:
        _HANN[(h, w)] = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return _HANN[(h, w)]


def overlap_score(ta: np.ndarray, tb: np.ndarray, min_overlap: float = 0.25) -> tuple[float, float]:
    """Do these two tiles show the same place? -> (correlation, overlap fraction).

    Phase correlation estimates the shift between the tiles; the overlapping part is
    then compared with a normalised cross-correlation. Unrelated tiles score ~0 even
    when their perceptual hashes happen to be close.

    Calibration (visual check of Airbus pairs, September 2026):
      >= 0.95  always the same place;  0.91-0.95 same coast/ship shifted (true);
      ~0.88    mixed (a coastline matched a cloud);  <= 0.83 mostly false.
    Hence the defaults 0.90 (accept) / 0.75 (ask ORB) in ``build_groups``.
    """
    a = ta.astype(np.float32)
    b = tb.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    win = _hann(*a.shape)
    (dx, dy), _ = cv2.phaseCorrelate(a * win, b * win)
    h, w = a.shape
    sx, sy = int(round(dx)), int(round(dy))
    if abs(sx) >= w or abs(sy) >= h:
        return 0.0, 0.0
    # cv2.phaseCorrelate(a, b) returns (dx, dy) with b(x, y) = a(x - dx, y - dy),
    # so pixel a[y, x] corresponds to b[y + dy, x + dx].
    ax0, bx0 = (0, sx) if sx >= 0 else (-sx, 0)
    ay0, by0 = (0, sy) if sy >= 0 else (-sy, 0)
    ow, oh = w - abs(sx), h - abs(sy)
    frac = (ow * oh) / (w * h)
    if frac < min_overlap:
        return 0.0, frac
    pa = a[ay0:ay0 + oh, ax0:ax0 + ow].ravel()
    pb = b[by0:by0 + oh, bx0:bx0 + ow].ravel()
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    denom = float(np.linalg.norm(pa) * np.linalg.norm(pb))
    return (float(pa @ pb / denom) if denom > 1e-6 else 0.0), frac


# No cheaper low-resolution screen: at 64 px a verified duplicate scored -0.30
# (0.93 at 128 px) because its detail only exists at full thumbnail resolution.

# Process-pool worker state: thumbnails are memory-mapped from disk, shared by the OS.
_W: dict = {}


def _worker_init(path128: str) -> None:
    _W["t128"] = np.load(path128, mmap_mode="r")
    cv2.setNumThreads(1)


def _worker_chunk(pairs: np.ndarray) -> np.ndarray:
    t128 = _W["t128"]
    out = np.zeros((len(pairs), 3), np.float32)
    for k, (a, b) in enumerate(pairs):
        out[k] = pair_evidence(t128[a], t128[b])
    return out


def verify_pairs_parallel(pairs: np.ndarray, path128: str, workers: int = 8,
                          chunk: int = 50_000, log=print) -> np.ndarray:
    """Overlap proof for many pairs using separate processes (no GIL contention).

    Measured: ~1 800 pairs/s per process, so ~3.5 min for 4.5 M pairs on 12 processes
    (threads were GIL-bound at ~420 pairs/s in total).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = np.zeros((len(pairs), 3), np.float32)       # coarse, detail, overlap
    bounds = [(lo, min(lo + chunk, len(pairs))) for lo in range(0, len(pairs), chunk)]
    with ProcessPoolExecutor(max(1, workers), initializer=_worker_init, initargs=(path128,)) as ex:
        futures = {ex.submit(_worker_chunk, pairs[lo:hi]): (lo, hi) for lo, hi in bounds}
        done = 0
        for fut in as_completed(futures):
            lo, hi = futures[fut]
            out[lo:hi] = fut.result()
            done += hi - lo
            log(f"      overlap proofs {done:,}/{len(pairs):,}")
    return out


_K5 = np.ones((5, 5), np.uint8)
_K7 = np.ones((7, 7), np.uint8)


def valid_mask(t: np.ndarray) -> np.ndarray:
    """True where the thumbnail shows real imagery, False on no-data fill + a margin.

    Airbus scene-edge tiles carry black or blue fill bars. They are perfectly flat
    (a 5x5 neighbourhood has min == max), whereas real sea always has some noise.
    The margin removes the sharp bar edge itself.
    """
    flat = cv2.dilate(t, _K5) == cv2.erode(t, _K5)
    return cv2.dilate(flat.astype(np.uint8), _K7) == 0


def pair_evidence(ta: np.ndarray, tb: np.ndarray, detail_sigma: float = 3.0,
                  min_overlap: float = 0.25) -> tuple[float, float, float]:
    """(coarse correlation, fine-detail correlation, valid overlap fraction) for two thumbnails.

    Lessons from the full Airbus run, each of which chained hundreds or thousands of
    unrelated tiles into one group:
      * HAZE: two smooth brightness gradients correlate whatever the scene
        -> agreement is also required on fine (high-pass) detail;
      * NO-DATA BARS: scene-edge tiles share black/blue fill bars whose sharp edge
        dominates both correlations on featureless sea
        -> fill pixels (+ margin) are excluded everywhere, including the shift estimate;
      * fine detail alone cannot find the shift (coastlines are coarse structure)
        -> the shift is estimated on the full, fill-neutralised image.
    Too little valid overlap, or a textureless one, gives detail = NaN: no evidence.
    """
    va, vb = valid_mask(ta), valid_mask(tb)
    a = ta.astype(np.float32)
    b = tb.astype(np.float32)
    h, w = a.shape
    if va.mean() < min_overlap or vb.mean() < min_overlap:
        return 0.0, float("nan"), 0.0
    a[~va] = a[va].mean()                     # neutralise fill: no edge, no signal
    b[~vb] = b[vb].mean()
    win = _hann(h, w)
    (dx, dy), _ = cv2.phaseCorrelate((a - a.mean()) * win, (b - b.mean()) * win)
    sx, sy = int(round(dx)), int(round(dy))
    if abs(sx) >= w or abs(sy) >= h:
        return 0.0, float("nan"), 0.0
    ax0, bx0 = (0, sx) if sx >= 0 else (-sx, 0)
    ay0, by0 = (0, sy) if sy >= 0 else (-sy, 0)
    ow, oh = w - abs(sx), h - abs(sy)
    m = va[ay0:ay0 + oh, ax0:ax0 + ow] & vb[by0:by0 + oh, bx0:bx0 + ow]
    frac = float(m.sum()) / (w * h)
    if frac < min_overlap:
        return 0.0, float("nan"), frac

    def ncc(pa: np.ndarray, pb: np.ndarray) -> float:
        pa = pa - pa.mean()
        pb = pb - pb.mean()
        den = float(np.linalg.norm(pa) * np.linalg.norm(pb))
        return float(pa @ pb / den) if den > 1e-6 else 0.0

    coarse = ncc(a[ay0:ay0 + oh, ax0:ax0 + ow][m], b[by0:by0 + oh, bx0:bx0 + ow][m])
    ha = a - cv2.GaussianBlur(a, (0, 0), detail_sigma)
    hb = b - cv2.GaussianBlur(b, (0, 0), detail_sigma)
    pa = ha[ay0:ay0 + oh, ax0:ax0 + ow][m]
    pb = hb[by0:by0 + oh, bx0:bx0 + ow][m]
    if pa.std() < 1.0 or pb.std() < 1.0:
        return coarse, float("nan"), frac
    return coarse, ncc(pa, pb), frac


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a: np.uint64, b: np.uint64) -> int:
    """Number of differing bits between two 64-bit hashes."""
    x = np.uint64(a) ^ np.uint64(b)
    return int(_POPCOUNT[np.frombuffer(np.array([x], dtype=">u8").tobytes(), dtype=np.uint8)].sum())


# --------------------------------------------------------------------------- union-find


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.rank[ra] += self.rank[ra] == self.rank[rb]
        return True

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = collections.defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return dict(out)


# --------------------------------------------------------------------------- candidates


def exact_hash_pairs(sigs: list[list[np.uint64]], max_bucket: int = 8) -> set[tuple[int, int]]:
    """Pairs whose whole-tile hash is identical (re-encoded / resaved copies).

    Buckets bigger than ``max_bucket`` are ignored: on blank open sea many unrelated
    tiles share a hash, and they would flood the verifier for nothing.
    """
    buckets: dict[int, list[int]] = collections.defaultdict(list)
    for idx, hashes in enumerate(sigs):
        if hashes:
            buckets[int(hashes[0])].append(idx)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if 1 < len(members) <= max_bucket:
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    pairs.add((a, b))
    return pairs


def candidate_pairs(sigs: list[list[np.uint64]], band_bits: int = 16, max_bucket: int = 200) -> set[tuple[int, int]]:
    """Image pairs sharing at least one exact band of one signature (LSH banding).

    NOT used by default: at 53 000 Airbus tiles it produced ~9.5 million candidate
    pairs (blank-sea buckets of up to 1 600 tiles) and exhausted memory, while the
    thumbnail nearest-neighbour search already finds the shifted crops we care about.
    """
    n_bands = 64 // band_bits
    mask = np.uint64((1 << band_bits) - 1)
    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for idx, hashes in enumerate(sigs):
        seen = set()
        for h in hashes:
            for b in range(n_bands):
                key = (b, int((np.uint64(h) >> np.uint64(b * band_bits)) & mask))
                if key not in seen:
                    seen.add(key)
                    buckets[key].append(idx)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if 1 < len(members) <= max_bucket:   # ignore degenerate buckets (e.g. blank tiles)
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    pairs.add((a, b) if a < b else (b, a))
    return pairs


def best_distance(sa: list[np.uint64], sb: list[np.uint64]) -> int:
    """Smallest Hamming distance over all window combinations of two images.

    Vectorised: one numpy pass instead of 36 scalar calls (matters at ~700k pairs).
    """
    a = np.asarray(sa, dtype=">u8").reshape(-1, 1)
    b = np.asarray(sb, dtype=">u8").reshape(1, -1)
    x = np.ascontiguousarray(a ^ b)
    return int(_POPCOUNT[np.frombuffer(x.tobytes(), dtype=np.uint8)].reshape(x.size, 8).sum(1).min())


# --------------------------------------------------------------------------- geometric check


@dataclass
class OrbVerifier:
    """Proves two tiles overlap: ORB keypoints + RANSAC homography inliers."""
    n_features: int = 600
    ratio: float = 0.75
    min_inliers: int = 40          # 12, then 25 were too permissive: on the full run, false
                                   # links had 25-32 inliers, true ones 48-71
    cache: dict = field(default_factory=dict, repr=False)

    def features(self, key: str, gray: np.ndarray | None = None):
        if key not in self.cache:
            orb = cv2.ORB_create(self.n_features)
            kp, des = orb.detectAndCompute(gray, None)
            self.cache[key] = (np.array([k.pt for k in kp], np.float32), des)
        return self.cache[key]

    def inliers(self, ka, da, kb, db) -> int:
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return 0
        src = np.float32([ka[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return int(mask.sum()) if mask is not None else 0


# --------------------------------------------------------------------------- grouping


def build_groups(ids: list[str], sigs: list[list[np.uint64]], vectors: np.ndarray | None = None,
                 thumbs: list[np.ndarray] | None = None, max_distance: int = 26,
                 coarse_accept: float = 0.90, detail_accept: float = 0.40, ncc_orb_low: float = 0.75,
                 verifier: OrbVerifier | None = None, gray_of=None, knn: int = 12,
                 max_orb_pairs: int = 3000, use_lsh: bool = True, lsh_bucket: int = 32,
                 workers: int = 8, thumb_path: str | None = None,
                 log=print) -> tuple[list[int], list[dict]]:
    """Return (group id per image, list of duplicate pairs with evidence).

    Candidates come from the hashes and the thumbnail neighbours; a pair is only
    accepted once the overlap is *proved* (see ``pair_evidence``):
      * coarse >= ``coarse_accept`` AND fine detail >= ``detail_accept``      -> "ncc"
      * else coarse >= ``ncc_orb_low`` and ORB + RANSAC has >= min_inliers   -> "orb"
    Both thresholds must be strict because of the BASE RATE: with ~4.5 million candidate
    pairs even a 0.1 % false-accept rate creates thousands of false links, and union-find
    chains them (coarse >= 0.6 produced one group of 6 644 unrelated tiles).
    ``max_distance`` only limits which candidates are worth verifying at all.
    """
    uf = UnionFind(len(ids))
    evidence: list[dict] = []
    candidates = exact_hash_pairs(sigs)
    if use_lsh:
        candidates |= candidate_pairs(sigs, max_bucket=lsh_bucket)
    if vectors is not None:
        candidates |= knn_pairs(vectors, k=knn)
    log(f"      candidates: {len(candidates):,} pairs")

    scored = []
    for a, b in candidates:
        if not sigs[a] or not sigs[b]:       # image failed to decode
            continue
        d = best_distance(sigs[a], sigs[b])
        if d <= max_distance:
            scored.append((d, a, b))
    scored.sort()
    del candidates
    log(f"      within hash distance {max_distance}: {len(scored):,} pairs to verify")

    # Overlap proof for every candidate pair: (coarse, detail, overlap) per pair.
    ev = np.zeros((len(scored), 3), np.float32)
    if scored and thumb_path is not None:
        # large runs: separate processes share memory-mapped thumbnails (threads were
        # GIL-bound: ~420 pairs/s, i.e. ~3 h for the 4.5 M pairs of the full dataset)
        pairs = np.array([(a, b) for _, a, b in scored], dtype=np.int32)
        ev = verify_pairs_parallel(pairs, thumb_path, workers, log=log)
    elif scored and thumbs is not None:
        for k, (_, a, b) in enumerate(scored):
            ev[k] = pair_evidence(thumbs[a], thumbs[b])

    orb_budget = max_orb_pairs
    for k, (d, a, b) in enumerate(scored):
        verdict, inl = None, None
        corr, detail, frac = float(ev[k, 0]), float(ev[k, 1]), float(ev[k, 2])
        if corr >= coarse_accept and detail >= detail_accept:      # NaN detail never passes
            verdict = "ncc"
        elif (verifier is not None and gray_of is not None and orb_budget > 0
              and corr >= ncc_orb_low):
            orb_budget -= 1
            ka, da = verifier.features(ids[a], gray_of(a))
            kb, db = verifier.features(ids[b], gray_of(b))
            inl = verifier.inliers(ka, da, kb, db)
            if inl >= verifier.min_inliers:
                verdict = "orb"
        if verdict:
            uf.union(a, b)
            evidence.append({"a": ids[a], "b": ids[b], "distance": d, "method": verdict,
                             "corr": round(corr, 3),
                             "detail": None if detail != detail else round(detail, 3),
                             "overlap": round(frac, 3), "inliers": inl})
    root_to_gid: dict[int, int] = {}
    group_of = []
    for i in range(len(ids)):
        r = uf.find(i)
        group_of.append(root_to_gid.setdefault(r, len(root_to_gid)))
    return group_of, evidence


# --------------------------------------------------------------------------- split


def group_aware_split(group_of: list[int], n_ships: list[int], ratios=(0.8, 0.1, 0.1),
                      seed: int = 0) -> list[str]:
    """Assign whole groups to train/val/test, balancing images *and* ships.

    Greedy: biggest groups first, each goes to the split that is furthest below its quota
    (measured on ships when the group has ships, otherwise on image count).
    """
    names = ("train", "val", "test")
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(group_of):
        groups[g].append(i)
    total_imgs = len(group_of)
    total_ships = sum(n_ships) or 1
    quota_i = {s: r * total_imgs for s, r in zip(names, ratios)}
    quota_s = {s: r * total_ships for s, r in zip(names, ratios)}
    have_i = dict.fromkeys(names, 0.0)
    have_s = dict.fromkeys(names, 0.0)
    rng = np.random.default_rng(seed)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), -sum(n_ships[i] for i in kv[1]),
                                                   rng.random()))
    out = [""] * total_imgs
    for _, members in order:
        ships = sum(n_ships[i] for i in members)
        if ships:
            target = max(names, key=lambda s: (quota_s[s] - have_s[s]) / max(1.0, quota_s[s]))
        else:
            target = max(names, key=lambda s: (quota_i[s] - have_i[s]) / max(1.0, quota_i[s]))
        have_i[target] += len(members)
        have_s[target] += ships
        for i in members:
            out[i] = target
    return out


def check_no_leakage(group_of: list[int], split: list[str]) -> list[int]:
    """Groups whose images landed in more than one split (must be empty)."""
    per_group: dict[int, set[str]] = collections.defaultdict(set)
    for g, s in zip(group_of, split):
        per_group[g].add(s)
    return [g for g, s in per_group.items() if len(s) > 1]
