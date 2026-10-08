"""sat7 -- IASTAM 6.0, Track 4, Problem 7: transmitting information rather than raw data.

Modules
-------
rle            Airbus Ship Detection run-length encoding <-> masks <-> boxes
prefilter      Classic computer-vision pre-filter (Otsu, morphology, watershed)
orbit          Ground-station contact windows and per-pass downlink capacity (+ multi-station)
scheduler      Multi-pass, value-aware downlink scheduler + baseline policies (LoD ladder)
b2_sahi_fusion B2: SAHI sliced inference + global-coordinate detection fusion (detector-agnostic)
relay          B4: inter-satellite visibility windows + direct-vs-relay path choice (SGP4-reused)

Paper-faithful additions (Phase 3)
----------------------------------
perception     First-class whole-image (B1) vs SAHI (B2) perception path + YOLO detect_fn + ablations
priority       Paper Table I P0-P3 semantic priority levels; a drop-in alternative to the LoD ladder
joint          The paper's joint program  min a*E + b*D + g*T  s.t. Accuracy>=A_min, InfoPres>=I_min
energy         Calibratable per-stage energy model (paper section V); inject measured edge P_k / T_k
datasets       Large-scene loaders (DOTA / YOLO / COCO-HRSID) + SAHI evaluation path by size bucket
"""
