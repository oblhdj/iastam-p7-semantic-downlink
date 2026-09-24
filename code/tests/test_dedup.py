import cv2
import numpy as np

from sat7.dedup import (OrbVerifier, UnionFind, best_distance, build_groups, check_no_leakage,
                        group_aware_split, hamming, knn_pairs, overlap_score, pair_evidence, phash,
                        signatures, thumb_gray, thumb_vector, valid_mask)
from sat7.synthetic import make_tile


def scene(seed: int, size: int = 1400):
    """A big 'satellite scene' from which we cut overlapping tiles."""
    img, _ = make_tile(n_ships=14, size=size, seed=seed)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_hamming_and_phash_basics():
    g = scene(1)
    assert hamming(phash(g), phash(g)) == 0
    assert hamming(np.uint64(0), np.uint64(0b1011)) == 3
    # JPEG compression must not change the fingerprint much
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    assert hamming(phash(g), phash(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE))) <= 4


def test_identical_and_unrelated_tiles():
    a, b = scene(2)[:768, :768], scene(3)[:768, :768]
    assert best_distance(signatures(a), signatures(a)) == 0
    assert best_distance(signatures(a), signatures(b)) > 8


def test_overlapping_crops_are_grouped():
    """The real Airbus problem: two tiles cut from the same scene, shifted by 120 px."""
    s = scene(4)
    tiles = {"a": s[0:768, 0:768], "b": s[120:888, 120:888], "c": scene(5)[0:768, 0:768]}
    ids = list(tiles)
    groups, evidence = _group(tiles, ids)
    assert groups[0] == groups[1], "shifted crops of the same scene must share a group"
    assert groups[2] != groups[0], "an unrelated tile must stay alone"
    assert evidence and evidence[0]["method"] in ("ncc", "orb")


def _group(tiles: dict, ids: list[str]):
    sigs = [signatures(tiles[i]) for i in ids]
    vecs = np.stack([thumb_vector(tiles[i]) for i in ids])
    thumbs = [thumb_gray(tiles[i]) for i in ids]
    return build_groups(ids, sigs, vecs, thumbs, verifier=OrbVerifier(),
                        gray_of=lambda k: tiles[ids[k]])


def test_similar_looking_but_unrelated_sea_tiles_are_not_grouped():
    """Regression: pHash alone matched unrelated open-sea tiles (distance 2!).

    Empty low-texture sea gives near-random hashes, so an overlap proof is required.
    """
    tiles = {f"t{i}": scene(50 + i)[0:768, 0:768] for i in range(6)}
    ids = list(tiles)
    groups, evidence = _group(tiles, ids)
    assert len(set(groups)) == len(ids), f"unrelated tiles were merged: {evidence}"


def _haze(seed: int, size: int = 768) -> np.ndarray:
    """A different hazy scene: smooth brightness gradient + independent fine noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 120 + 60 * (xx / size) + 30 * (yy / size)
    return np.clip(base + rng.normal(0, 6, (size, size)), 0, 255).astype(np.uint8)


def test_hazy_scenes_are_not_merged():
    """Regression (full Airbus run): haze gradients chained 440 unrelated tiles."""
    tiles = {f"h{i}": _haze(100 + i) for i in range(5)}
    ids = list(tiles)
    coarse, detail, _ = pair_evidence(thumb_gray(tiles["h0"]), thumb_gray(tiles["h1"]))
    assert coarse > 0.6, "the trap: coarse correlation looks convincing"
    assert detail < 0.4, "fine detail must reveal they are different scenes"
    groups, evidence = _group(tiles, ids)
    assert len(set(groups)) == len(ids), f"hazy tiles were merged: {evidence}"


def test_nodata_bars_are_not_evidence():
    """Regression (full Airbus run): scene-edge tiles with black/blue fill bars on
    featureless sea formed groups of 75-214 unrelated tiles."""
    rng = np.random.default_rng(7)
    tiles = {}
    for i in range(5):
        sea = np.clip(60 + 8 * i + rng.normal(0, 3, (768, 768)), 0, 255).astype(np.uint8)
        ys = (560 + 0.08 * np.arange(768)).astype(int)            # same slanted bar edge
        for x, y0 in enumerate(ys):
            sea[y0:, x] = 0                                        # black fill
        sea[700:, :] = 29                                          # "blue" fill in grey
        tiles[f"n{i}"] = sea
    ids = list(tiles)
    groups, evidence = _group(tiles, ids)
    assert len(set(groups)) == len(ids), f"no-data bars merged tiles: {evidence}"


def test_valid_mask_marks_fill_and_keeps_sea():
    rng = np.random.default_rng(0)
    sea = np.clip(80 + rng.normal(0, 4, (128, 128)), 0, 255).astype(np.uint8)
    sea[100:, :] = 0
    v = valid_mask(sea)
    assert v[:80].mean() > 0.95 and v[105:].mean() == 0.0


def test_overlap_score_separates_same_scene_from_different_scenes():
    s = scene(7)
    a, b = thumb_gray(s[0:768, 0:768]), thumb_gray(s[100:868, 100:868])
    c = thumb_gray(scene(8)[0:768, 0:768])
    corr_same, frac = overlap_score(a, b)
    corr_diff, _ = overlap_score(a, c)
    assert corr_same > 0.8 and frac > 0.5
    assert corr_diff < 0.45
    assert overlap_score(a, a)[0] > 0.99


def test_knn_finds_the_shifted_crop_among_distractors():
    s = scene(6)
    tiles = [s[0:768, 0:768], s[60:828, 60:828]] + [scene(10 + i)[0:768, 0:768] for i in range(8)]
    vecs = np.stack([thumb_vector(t) for t in tiles])
    assert (0, 1) in knn_pairs(vecs, k=3)


def test_union_find():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2) != uf.find(3)
    assert sorted(len(v) for v in uf.groups().values()) == [1, 1, 3]


def test_split_keeps_groups_together_and_balances():
    rng = np.random.default_rng(0)
    group_of = [i // 3 for i in range(300)]              # 100 groups of 3 tiles
    n_ships = [int(rng.integers(0, 4)) for _ in range(300)]
    split = group_aware_split(group_of, n_ships, seed=1)
    assert check_no_leakage(group_of, split) == []
    share = {s: split.count(s) / len(split) for s in ("train", "val", "test")}
    assert 0.7 <= share["train"] <= 0.9
    assert all(share[s] > 0.02 for s in ("val", "test"))
    for s in ("train", "val", "test"):                   # every split must contain ships
        assert sum(n for n, sp in zip(n_ships, split) if sp == s) > 0
