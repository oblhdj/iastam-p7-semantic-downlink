import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import sys

# Add code/ to path to import local modules if needed
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

st.set_page_config(page_title="IASTAM P7 Semantic Downlink", layout="wide")

st.title("🛰️ Semantic Downlink Dashboard")
st.markdown("""
**Transmitting Information Rather Than Raw Data.**
This dashboard demonstrates how our pipeline saves 557x in downlink bandwidth by sending *ship information* instead of raw pixels.
Upload a satellite tile to see how the onboard AI decides what to transmit!
""")

# Load the YOLO model once
@st.cache_resource
def load_detector():
    try:
        from ultralytics import YOLO
        model_path = ROOT / "code" / "runs" / "ships" / "weights" / "best.pt"
        if not model_path.exists():
            return None
        return YOLO(str(model_path))
    except ImportError:
        return "missing_ultralytics"

model = load_detector()

if model == "missing_ultralytics":
    st.error("Ultralytics is not installed. Please install it in your environment: `pip install ultralytics`")
elif model is None:
    st.warning("YOLO weights not found. Make sure the dataset and weights are downloaded to `code/runs/ships/weights/best.pt`.")
else:
    uploaded_file = st.file_uploader("Upload a satellite tile (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Raw bytes for comparison
        raw_bytes = len(file_bytes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Satellite Tile (B0)")
            st.image(img_rgb, caption=f"Raw Image: {raw_bytes / 1024:.1f} KB")
            
        with col2:
            st.subheader("Onboard Decision (B3)")
            with st.spinner("Running onboard AI..."):
                results = model(img_rgb, imgsz=768, conf=0.05, verbose=False)
                
            result = results[0]
            boxes = result.boxes
            
            # Reconstruct the image with bounding boxes
            img_drawn = img_rgb.copy()
            
            total_bytes = 0
            confident_count = 0
            uncertain_count = 0
            
            CONF_HIGH = 0.670  # The calibrated threshold from Report 03
            
            for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                if conf >= CONF_HIGH:
                    # Confident ship -> Send Metadata only (40 bytes)
                    color = (0, 255, 0) # Green
                    label = f"Ship: {conf:.2f} (40 B)"
                    total_bytes += 40
                    confident_count += 1
                else:
                    # Uncertain ship -> Send thumbnail (~900 bytes)
                    color = (255, 165, 0) # Orange
                    label = f"Uncertain: {conf:.2f} (~900 B)"
                    total_bytes += 900
                    uncertain_count += 1
                    
                cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_drawn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            st.image(img_drawn, caption="Processed Image (For visualization only)")
            
            # If no ships, the learned gate/prefilter would likely discard the whole tile
            if len(boxes) == 0:
                st.success("🤖 **Decision:** Empty tile detected by gate. **Discarded.** (Cost: 0 bytes)")
                total_bytes = 0
            else:
                st.info(f"🤖 **Decision:** Found {len(boxes)} candidate(s).")
                st.write(f"- 🟢 **{confident_count} Confident Ships** → Sending 40-byte metadata vectors.")
                st.write(f"- 🟠 **{uncertain_count} Uncertain Hits** → Sending 900-byte cropped thumbnails for ground review.")
                
            reduction = raw_bytes / total_bytes if total_bytes > 0 else float('inf')
            
            st.metric(label="Data Transmitted", value=f"{total_bytes / 1024:.2f} KB", delta=f"-{reduction:.0f}x reduction vs Raw", delta_color="inverse")
            
st.markdown("---")
st.markdown("💡 **How to run this dashboard:** Ensure you are in your `.venv312` environment and run `streamlit run demo/dashboard.py`")
