import numpy as np

from sat7.rle import Box, mask_to_box, rle_decode, rle_encode, rles_to_boxes


def test_roundtrip_random_mask():
    rng = np.random.default_rng(0)
    mask = (rng.random((768, 768)) > 0.97).astype(np.uint8)
    assert np.array_equal(rle_decode(rle_encode(mask)), mask)


def test_column_major_convention():
    # Airbus numbers pixels top-to-bottom first: "1 3" = first 3 pixels of column 0.
    m = rle_decode("1 3", (4, 4))
    assert m[:3, 0].tolist() == [1, 1, 1] and m.sum() == 3


def test_empty_rle_gives_empty_mask():
    assert rle_decode("", (8, 8)).sum() == 0
    assert rle_decode(float("nan"), (8, 8)).sum() == 0
    assert rle_encode(np.zeros((8, 8), np.uint8)) == ""


def test_boxes_and_yolo():
    mask = np.zeros((768, 768), np.uint8)
    mask[100:110, 200:230] = 1                       # 10 px tall, 30 px wide
    box = mask_to_box(mask)
    assert box == Box(200, 100, 30, 10)
    cx, cy, w, h = box.to_yolo(768, 768)
    assert abs(cx - 215 / 768) < 1e-9 and abs(cy - 105 / 768) < 1e-9
    assert rles_to_boxes([rle_encode(mask), ""]) == [box]


def test_iou():
    a, b = Box(0, 0, 10, 10), Box(5, 0, 10, 10)
    assert abs(a.iou(b) - 50 / 150) < 1e-9
    assert Box(0, 0, 2, 2).iou(Box(5, 5, 2, 2)) == 0.0
