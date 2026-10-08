import numpy as np

from sat7.datasets import (Scene, evaluate_scenes, parse_dota_lines, parse_yolo_lines,
                           recall_by_size, recall_of, scenes_from_coco)
from sat7.perception import PerceptionConfig


def test_parse_dota_skips_headers_and_filters_class():
    lines = ["imagesource:GE", "gsd:0.3",
             "100 100 140 100 140 160 100 160 ship 0",
             "0 0 10 0 10 10 0 10 harbor 0"]
    boxes = parse_dota_lines(lines, classes=("ship",))
    assert len(boxes) == 1
    cx, cy, w, h = boxes[0]
    assert (cx, cy, w, h) == (120.0, 130.0, 40.0, 60.0)


def test_parse_dota_all_classes_when_none():
    lines = ["5 5 15 5 15 15 5 15 plane 0"]
    assert len(parse_dota_lines(lines, classes=None)) == 1


def test_parse_yolo_denormalises_and_filters_class():
    lines = ["0 0.5 0.5 0.1 0.2", "3 0.1 0.1 0.05 0.05"]
    boxes = parse_yolo_lines(lines, 1000, 1000, classes={0})
    assert boxes == [(500.0, 500.0, 100.0, 200.0)]


def test_scenes_from_coco_maps_bbox_to_centre_form():
    coco = {"categories": [{"id": 1, "name": "ship"}, {"id": 2, "name": "plane"}],
            "images": [{"id": 7, "file_name": "a.png", "width": 800, "height": 600}],
            "annotations": [{"image_id": 7, "category_id": 1, "bbox": [10, 20, 40, 60]},
                            {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]}]}
    scenes = scenes_from_coco(coco, category_names=("ship",))
    assert len(scenes) == 1 and scenes[0].gt_boxes == [(30.0, 50.0, 40.0, 60.0)]
    assert scenes[0].width == 800 and scenes[0].height == 600


def test_recall_of_is_greedy_one_to_one():
    gts = [(100, 100, 20, 20), (500, 500, 20, 20)]
    dets = [(100, 100, 20, 20)]                     # only the first ship detected
    assert recall_of(dets, gts) == [True, False]


def test_recall_by_size_buckets():
    found = [True, False, True]
    sizes = [10, 50, 200]                           # small, medium, large
    r = recall_by_size(found, sizes)
    assert r["small_<32"] == 1.0 and r["medium"] == 0.0 and r["large_>96"] == 1.0 and r["n_gt"] == 3


def test_evaluate_scenes_runs_sahi_path():
    # stub detector that, in each crop, reports a box at the crop centre -> exercises slicing + fusion
    sc = Scene("t", 1536, 768, gt_boxes=[(200, 200, 40, 40)])

    def detect(crops):
        return [[(c.shape[1] / 2, c.shape[0] / 2, 30, 30, 0.9)] for c in crops]
    out = evaluate_scenes([sc], lambda s: np.zeros((s.height, s.width, 3), np.uint8),
                          detect, PerceptionConfig(mode="sahi", window=768))
    assert out["scenes"] == 1 and out["mode"] == "sahi" and out["recall"]["n_gt"] == 1
