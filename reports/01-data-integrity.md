# WP0 -- Data integrity report

Source: `airbus-ship-detection.zip` (192556 training tiles in the zip).

## Selection
53195 tiles: 42556 with ships + 10639 empty
(ratio 0.25), 81723 ship instances.

## Near-duplicate detection
Perceptual hash (pHash, 64 bit) over 6 windows per tile, LSH banding for candidates,
Hamming distance <= 26 accepted directly, ambiguous pairs confirmed by ORB + RANSAC.

* duplicate pairs found: **6282** {'ncc': 4823, 'orb': 1459}
* groups: **47313** (4259 contain duplicates, largest 63 tiles)
* tiles inside a duplicate group: **10141 (19.1%)**

## Leakage-free split
Whole groups are assigned to one split, balancing tiles and ships.

| split | tiles | ships |
|---|---|---|
| train | 42555 | 65377 |
| val | 5320 | 8173 |
| test | 5320 | 8173 |

* groups spanning several splits: **0** (must be 0)
* a naive random split would have leaked **1617** groups

Runtime 2110.4 s. Settings: {'max_distance': 26, 'orb': True, 'empty_ratio': 0.25, 'seed': 0}.
