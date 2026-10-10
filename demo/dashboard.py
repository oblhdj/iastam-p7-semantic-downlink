"""Streamlit view of the onboard perception stage on one image (upload one, or pick a bundled tile).

    streamlit run demo/dashboard.py

It is a thin UI over the same sat7 functions scripts/detect_image.py uses -- input validation
(sat7.imagery), the configured detector (sat7.perception.load_detector: the ONNX export on CPU, no
torch needed), whole-image or SAHI detection with global-coordinate fusion (detect_image), and
evaluation only when ground truth is supplied (sat7.evaluation). Fixed per audit H6: the image goes
to the detector in BGR (it went in as RGB), the 0.25 onboard cut decides what counts as a detection
(every box >= 0.05 was counted), "raw" is H x W x 3 bytes as in B0 (it was the JPEG's file size),
and no gate decision or headline number is claimed that this page does not compute. Semantic
packet costing (P0-P3 with the WP4-measured sizes) is demonstrated in demo/quickstart, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from sat7.datasets import parse_yolo_lines  # noqa: E402
from sat7.evaluation import describe_detections, evaluate_detections  # noqa: E402
from sat7.imagery import decode_image  # noqa: E402
from sat7.perception import PerceptionConfig, detect_image, load_detector, slice_count  # noqa: E402

WEIGHTS = (ROOT / "code" / "runs" / "ships" / "weights" / "best.onnx",
           ROOT / "demo" / "quickstart" / "model" / "best.onnx",
           ROOT / "code" / "runs" / "ships" / "weights" / "best.pt")
SAMPLES = sorted((ROOT / "demo" / "quickstart" / "tiles").glob("*.jpg"))

st.set_page_config(page_title="IASTAM P7 perception", layout="wide")
st.title("Onboard perception: what the detector finds in one image")
st.caption("YOLOv8n trained on Airbus tiles; SAHI + global-coordinate fusion for images larger "
           "than one 768 px window. Bundled samples are synthetic stand-ins, not Airbus imagery.")


@st.cache_resource
def get_detector():
    w = next((p for p in WEIGHTS if p.exists()), None)
    return (load_detector(w), w) if w else (None, None)


det, weights = get_detector()
if det is None:
    st.error("No detector weights found (code/runs/ is not in git; demo/quickstart/model/best.onnx "
             "is the bundled CPU copy).")
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    conf = st.slider("Operating confidence cut", 0.05, 0.95, 0.25, 0.05,
                     help="0.25 is the onboard threshold every experiment uses")
    mode = st.selectbox("Mode", ["auto", "whole", "sahi"],
                        help="auto = one window if the image fits 768 px, SAHI otherwise")
    window = st.number_input("SAHI window (px)", 128, 2048, 768, 64)
    overlap = st.slider("SAHI overlap", 0.0, 0.5, 0.20, 0.05)
    st.caption(f"Model: {weights.name}")

src = st.radio("Image", ["bundled sample", "upload"], horizontal=True)
data, name = None, None
if src == "bundled sample":
    pick = st.selectbox("Sample tile", [p.name for p in SAMPLES]) if SAMPLES else None
    if pick:
        data, name = (ROOT / "demo" / "quickstart" / "tiles" / pick).read_bytes(), pick
else:
    up = st.file_uploader("Satellite image (JPEG / PNG / TIFF / BMP, 8-bit)",
                          type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    if up is not None:
        data, name = up.getvalue(), up.name
labels = st.file_uploader("Optional ground truth (YOLO .txt: class cx cy w h, normalised)", type=["txt"])

if data is not None:
    try:
        img, info = decode_image(data, name)
    except ValueError as e:
        st.error(f"Image rejected: {e}")
        st.stop()
    m = mode if mode != "auto" else ("whole" if max(info.width, info.height) <= window else "sahi")
    cfg = PerceptionConfig(mode=m, window=int(window), overlap=float(overlap))
    dets = detect_image(img, det, cfg)                     # BGR in, global (cx, cy, w, h, conf) out
    kept = sorted((d for d in dets if d[4] >= conf), key=lambda d: -d[4])

    shown = img.copy()
    for (cx, cy, w, h, c) in sorted(dets, key=lambda d: d[4]):
        p0, p1 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        if c >= conf:
            cv2.rectangle(shown, p0, p1, (60, 220, 60), 2)
            cv2.putText(shown, f"ship {c:.2f}", (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(shown, p0, p1, (150, 150, 150), 1)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
        st.write(f"**{info.width} x {info.height} px**, {info.format}, {info.channels_in} channel(s)")
        st.write(f"File on disk: **{info.file_bytes:,} B** · raw (H x W x 3, the B0 definition): "
                 f"**{info.raw_bytes:,} B**")
        for n in info.notes:
            st.caption(f"note: {n}")
    with c2:
        st.subheader(f"Detections ({m}, {slice_count(info.width, info.height, cfg)} detector call(s))")
        st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width="stretch")
        st.write(f"**{len(kept)}** detections at or above {conf} (green); "
                 f"{len(dets) - len(kept)} below the cut (grey).")
        if kept:
            st.dataframe([{"class": "ship", "conf": round(d[4], 3), "cx": round(d[0], 1),
                           "cy": round(d[1], 1), "w": round(d[2], 1), "h": round(d[3], 1)} for d in kept])

    st.subheader("Evaluation")
    if labels is not None:
        gts = parse_yolo_lines(labels.getvalue().decode().splitlines(), info.width, info.height, classes={0})
        ev = evaluate_detections([(dets, gts)], conf_thr=conf)
        st.write(f"{len(gts)} ground-truth ships · TP {ev['TP']} · FP {ev['FP']} · FN {ev['FN']}")
        st.write(f"precision **{ev['precision']}** · recall **{ev['recall']}** · F1 **{ev['F1']}** · "
                 f"AP50 {ev['AP50']} · AP50-95 {ev['AP50-95']}")
        st.caption("One image is a spot check, not a benchmark; the measured figures are in code/results/.")
    else:
        d = describe_detections(dets, conf)
        st.info("No ground truth supplied: counts and confidences only. These are NOT accuracy metrics.")
        c = d[f"confidence_at_or_above_{conf}"]
        st.write(f"{d['n_at_or_above_thr']} detections ≥ {conf}"
                 + (f"; confidence min {c['min']} / median {c['median']} / max {c['max']}" if c else ""))
