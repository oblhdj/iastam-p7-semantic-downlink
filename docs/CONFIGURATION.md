# Configuration

Every setting of the pipeline is a field of a Python dataclass in `code/sat7/`, passed to the
function that uses it. There is no global configuration file and no hidden state: a script builds
the configuration objects it needs, usually from its command-line flags, and the result files record
the values that mattered. This page lists every one of them with its default.

The tables in the second half are generated from the code (`python code/scripts/print_config.py`)
and a test fails if they drift, so a default shown here is the default in the source.

## Operating values that differ from a class default

A class default is what you get from `SomeConfig()`. Three settings are overridden by every
experiment, and the value to quote is the operating one.

| setting | class default | operating value | where it is set |
|---|---|---|---|
| onboard detection cut | `LoDConfig.conf_low = 0.4` | **0.25** | every `wp` script passes `LoDConfig(conf_low=det_thr)`, `--det-thr 0.25`; `EncoderConfig.det_thr` and `RealWorkloadConfig.det_thr` already default to 0.25 |
| raw detector dump | 0.25 (ultralytics) | **0.05** | `sat7.perception.DEFAULT_RAW_CONF`: boxes are kept down to 0.05 so that AP has a curve; the 0.25 cut decides what counts |
| crop sizes | `LoDConfig.size_model = None` (flat `l1_bytes` / `l2_bytes`) | the fitted power law | `load_size_model()` reads `code/results/wp6_size_model.json` |

The P1 / P2 boundary is **0.670** in both schemes (`LoDConfig.conf_high`, `PriorityConfig.p1_conf`).
It comes from the calibration in report 03 and replaced an earlier 0.9.

## How to change a setting

* **In a script:** `python code/scripts/<name>.py --help` lists its flags. The flags behind every
  committed result are the lines of `code/rerun_all.sh`; that file is the record.
* **In code:** build the dataclass with the field you want, for example
  `PerceptionConfig(mode="sahi", window=512)` or `CommsConfig(link_share=0.35)`, and pass it in.
* **In the dashboard:** the sidebar sets the six fields of `demo_pipeline.Settings` (last table
  below). Everything else on the page runs at the defaults listed here.

A changed setting changes results. The committed numbers correspond to the defaults and operating
values on this page, and report 11 sweeps the invented ones.

## Environment variables and files

| name | effect |
|---|---|
| `P7_DEMO_FORCE_FALLBACK=1` | the dashboard does not load the live pipeline and shows `demo/fallback_assets/` (same as `python demo/launch.py --fallback`) |
| `P7_DEMO_ALLOW_THROTTLING=1` | the demo processes do not opt out of Windows power throttling (used to measure its effect; see `demo/DASHBOARD.md`) |
| `demo/.streamlit/config.toml` | Streamlit start-up settings for the dashboard: no first-run email prompt, listen on localhost only, no usage statistics |
| `demo/quickstart/model/best.onnx` | the detector the demos load. `code/runs/ships/weights/best.onnx` is used when the tracked copy is absent |
| `demo/quickstart/tiles/manifest.json` | the nine bundled synthetic tiles and their checksums |
| `code/results/` | read by the demos and by `sat7.real_workload`; never written by a demo |

## Labels on a setting

The comments quoted in the tables use the repository's labels. **ASSUMPTION** marks a value nobody
measured (powers, link share, orbit mix, values of products); **MEASURED** / **REAL** marks one that
came from the data. The comments are quoted from the source as they stand. Where a comment cites a
measurement, `code/results/` is authoritative: one of them still says the coastal tile is "62% of
the budget", the share before the tiles were measured; it is 76% today (`wp29_validation.json`).

## Every configuration object

<!-- BEGIN GENERATED: python code/scripts/print_config.py -->

### Perception

#### `sat7.perception.PerceptionConfig`

How the detector sees one frame: B1 (`whole`) or B2 (`sahi`).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `mode` | `'whole'` |  |
| `window` | `768` | SAHI slice size (px). 768 = detector-native (wp16 default); 512 = paper |
| `overlap` | `0.2` | neighbour-window overlap fraction; 0.0 = "without overlap" ablation |
| `nms_iou` | `0.5` | IoU above which two detections merge in fusion (highest conf wins) |
| `min_conf` | `0` | drop detections below this before fusing |
| `fuse` | `True` | False = "without detection fusion" ablation (duplicates survive) |

#### `sat7.b2_sahi_fusion.SahiConfig`

The window grid and fusion rule `PerceptionConfig` drives.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `window` | `768` | slice size in px; 768 = detector-native (recommended), 512 = paper |
| `overlap` | `0.2` | fraction of overlap between neighbouring windows |
| `nms_iou` | `0.5` | IoU above which two fused detections are merged (highest conf wins) |
| `min_conf` | `0` | drop detections below this before fusing (detector usually pre-cuts) |

### Pre-filter

#### `sat7.prefilter.PrefilterConfig`

The classic CV stage that supplies each tile's context (cloud / coast / sea).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `blur_ksize` | `3` | Gaussian denoising before everything else |
| `tophat_ksize` | `31` | must be larger than the biggest ship (px) |
| `min_contrast` | `12` | floor on the top-hat threshold (grey levels) |
| `k_sigma` | `4` | threshold >= mean + k * std of the top-hat |
| `open_ksize` | `3` | opening kernel: removes 1-2 px wave speckle |
| `min_area` | `12` | smallest blob kept as a ship candidate (px) |
| `max_area` | `6000` | bigger blobs are land / cloud, not a ship |
| `watershed_min_area` | `400` | only try to split blobs bigger than this |
| `cloud_v_min` | `170` | HSV value (brightness) of cloud pixels |
| `cloud_s_max` | `40` | HSV saturation of cloud pixels |
| `cloud_frac` | `0.9` | drop the whole tile only above this fraction; below it, clouds are masked and the clear sea searched |
| `cloud_margin` | `9` | px of dilation around clouds (their edges look like objects) |
| `cloud_min_size` | `61` | opening kernel: bright regions smaller than this are ships (white too!), not clouds |
| `land_delta` | `45` | land = brighter than the sea median by this much |
| `land_blob_frac` | `0.08` | tile is "land_coast" if big blobs cover this |
| `edge_density_land` | `0.12` | ... or if Canny edges cover this fraction |

### Semantic packet

#### `sat7.semantic.EncoderConfig`

What is sent for a detection and how large it is (records, ROI and context crops, packets).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `det_thr` | `0.25` | below: P0, never transmitted |
| `priority` | `PriorityConfig(...)` |  |
| `roi_margin` | `0.15` | WP4 "tight" crop  (P2 ROI = the L1 chip) |
| `roi_quality` | `40` |  |
| `ctx_margin` | `1.5` | WP4 "wake" crop   (P3 context away from a coast) |
| `ctx_quality` | `80` |  |
| `coast_tile_quality` | `40` | whole tile        (P3 context on a coast, WP7 q40) |
| `progressive` | `True` | truncatable images (the scheduler's progressive items) |
| `metadata` | `'record'` | "record": one packet per P1 record (finest scheduling) "tile":   one packet per tile, image id / time once |
| `max_packet_data` | `4096` |  |
| `class_id` | `0` |  |
| `dark_weight` | `5` | value of a dark vessel's report (LoDConfig.dark_weight) |

#### `sat7.priority.PriorityConfig`

The P0-P3 policy of the paper's Table I.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `p1_conf` | `0.67` | >= this: confident -> P1 metadata only (== LoDConfig.conf_high) |
| `coast_to_p3` | `True` | coastal detections escalate to P3 (which ones: coast_escalation) |
| `coast_escalation` | `'all'` | "all": every coastal detection (the measured default) \| "uncertain": only coastal detections below p1_conf |
| `dark_bump` | `True` | a dark (no-AIS) vessel escalates one level (capped at P3) |
| `roi_value` | `1` | a verified crop is worth ~one ship of extra evidence |
| `context_value` | `0.5` | the surrounding context adds less again (diminishing returns) |
| `coast_context_tile` | `True` |  |
| `gate_discard` | `True` | honour Workload.gate_empty: a gated-empty tile is pure P0 (no audit thumbnail -- that is a LoD safety net the P-scheme replaces with P3 context on uncertain real detections) |

#### `sat7.semantic.TMFraming`

CCSDS TM link layer, used only to report on-air bytes.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `frame_len` | `1115` |  |
| `tf_header` | `6` |  |
| `fecf` | `2` |  |
| `asm` | `4` |  |
| `rs_parity` | `160` |  |

### Level of detail and scheduling

#### `sat7.scheduler.LoDConfig`

The level-of-detail ladder behind the headline (sizes, thresholds, values).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `l0_bytes` | `40` | ASSUMPTION: lat, lon, length, heading, confidence. The serialized record (sat7.semantic) is 26 B + 6 B packet header + 2 B CRC = 34 B, measured |
| `l1_bytes` | `900` | small image chip      (measured: tight crop, q40) |
| `l2_bytes` | `2500` | ROI incl. wake        (measured: wake crop, q80) |
| `coast_tile_bytes` | `None` | compressed coastal / port tile (JPEG q40). None = MEASURED: the tile's own measured size when the workload carries one (Workload.coast_bytes, real days), else the measured mean COAST_TILE_BYTES_MEASURED. A number = that flat size for every tile (a sweep, or COAST_TILE_BYTES_WP7 to reproduce the earlier modeled estimate). See coast_tile_size(). History: WP7 moved this from q60 to q40 on its whole-tile ladder (42.4 -> 32.4 kB) -- sizes that describe open-sea tiles, about half a real coastal tile (10 Oct 2026). q40 is NOT free: report 14 re-ran the detector on recompressed coastal tiles and it costs 4.0 points of recall, all on small ships; q30 was rejected (-5.4 points). |
| `coast_mode` | `'tile'` | "tile"    = whole coastal tile, catches ships the detector missed (62% of the budget) "mosaic"  = ROI crops around detected ships only (WP7: 2.3 kB/tile, 14x cheaper, but it gives up the undetected-ship safety net) "adaptive"= the level-of-detail idea applied to the tile itself: mosaic normally, whole tile only where the cheap classic stage saw objects the network did not confirm |
| `coast_mosaic_bytes` | `2259` | measured: ROI mosaic x1.5 q60, 400 real tiles |
| `coast_escalate` | `11` | "adaptive": unconfirmed candidates needed to send the whole tile. MEASURED on the real coastal tiles: at 11+ (50% of them) sit 75% of the ships the network missed, while tiles with none hold 5%. |
| `esc_min` | `0` | plenty of capacity: every coastal tile goes whole |
| `esc_max` | `40` | hopeless backlog: effectively mosaic-only |
| `pressure_lo` | `0.9` | queued bytes / capacity before the queue can drain |
| `pressure_hi` | `3.5` |  |
| `thumb_bytes` | `1000` | safety-net thumbnail  (measured: 96 px q60 984 B; WP7: 128 px q30 is 1,037 B and *more* ships stay visible, 0.643 vs 0.615 -- resolution beats quality here because the JPEG header floor is ~700 B) |
| `thumb_value` | `0.01` |  |
| `conf_high` | `0.67` |  |
| `conf_low` | `0.4` |  |
| `dark_weight` | `5` |  |
| `size_model` | `None` |  |
| `dark_mode` | `'decoupled'` |  |
| `dark_wake_value` | `1` | ASSUMPTION: the wake/context is worth about one ordinary ship of extra evidence, on top of the report that the base chip already delivered |
| `gate_thumbnails` | `True` | skip the thumbnail when the onboard gate calls the tile empty (needs Workload.gate_empty; a synthetic workload has none, so nothing changes there). WP7: -3.6% bytes, no measured recall cost. Small because the classic pre-filter clears only 21% of empty tiles -- a learned gate (WP3) would raise it. |
| `thumb_audit` | `0.02` | fraction of gated-empty tiles thumbnailed anyway, so silent domain shift stays detectable |

#### `sat7.scheduler.SizeModel`

The measured power law for crop sizes (fitted by wp6_fit_size_model.py).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `l1_a` | `137.5` | crop tight q40 : R2 0.76 |
| `l1_b` | `0.57` |  |
| `l2_a` | `43.3` | crop wake q80  : R2 0.89 |
| `l2_b` | `1.14` |  |
| `min_bytes` | `120` | JPEG header floor |

#### `sat7.real_workload.RealWorkloadConfig`

The simulated day built from the detection catalogue.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `hours` | `24` |  |
| `tiles_per_day` | `40000` | ASSUMPTION: duty cycle of the imager |
| `mix` | `'orbit'` | "orbit" (assumed mix) \| "dataset" (file's own mix) |
| `orbit_mix` | `(0.15, 0.2, 0.05, 0.6)` | ASSUMPTION |
| `p_dark` | `0.1` | ASSUMPTION: no AIS in the Airbus data |
| `det_thr` | `0.25` | onboard detection threshold (real conf compared to it) |
| `seed` | `0` |  |

### Orbit and links

#### `sat7.orbit.OrbitConfig`

The primary satellite's orbit.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `altitude_km` | `500` |  |
| `inclination_deg` | `97.4` | ~sun-synchronous at 500 km |
| `raan_deg` | `0` |  |
| `epoch` | `2026-09-18 00:00 UTC` |  |

#### `sat7.orbit.GroundStation`

The ground station.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `name` | `'Sfax (ENIS)'` |  |
| `lat_deg` | `34.74` |  |
| `lon_deg` | `10.76` |  |
| `min_elevation_deg` | `5` |  |

#### `sat7.orbit.LinkConfig`

The ground-link budget model (preset `cubesat_sband`).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `name` | `'cubesat_sband'` |  |
| `bandwidth_hz` | `2e+06` | occupied bandwidth |
| `snr_zenith_db` | `12` | SNR when the satellite is straight overhead |
| `max_rate_bps` | `8e+06` | modem limit |
| `efficiency` | `0.8` | coding / protocol overhead (0..1) |

#### `sat7.comms.CommsConfig`

The two-path (direct / relay) link simulator: shares, rates, powers, routing.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `link` | `LinkConfig(...)` |  |
| `link_share` | `0.25` | share of each primary ground pass given to this payload |
| `relay_link_share` | `None` | share of each relay ground pass given to forwarded data (None = the primary's share) |
| `p_tx_W` | `15` | ground transmitter, both satellites (report 17) |
| `isl_rate_bps` | `2.53e+06` | raw ISL rate (report 19) |
| `isl_efficiency` | `0.8` | coding / protocol overhead, as LinkConfig.efficiency |
| `isl_share` | `1` | share of each ISL window given to this payload |
| `p_isl_W` | `12` | ISL transmitter (report 19) |
| `isl_setup_s` | `0` | acquisition / pointing time lost at the start of a window |
| `isl_grazing_km` | `100` | line of sight must clear Earth by this much |
| `isl_max_range_km` | `None` | optional range limit |
| `relay_raan_deg` | `90` | the relay's orbital plane (complementary coverage) |
| `ground_availability` | `1` |  |
| `isl_availability` | `1` |  |
| `availability_seed` | `0` |  |
| `relay_enabled` | `True` |  |
| `lam_T` | `0.000277778` | J = lam_E * E[J] + lam_T * T[s] |
| `lam_E` | `0` |  |
| `deadline_s` | `None` | latency constraint applied to every item (None = none) |
| `relay_kinds` | `('P1', 'P2', 'P3', 'L0', 'L1', 'L2', 'tile', 'thumb', 'mosaic', 'patch', 'fp')` | what the relay may forward: processed products only |
| `relay_storage_bytes` | `8e+09` |  |

#### `sat7.relay.RelayConfig`

The earlier window-only relay model (the canonical B4 row).

| field | default | meaning (the comment in the source) |
|---|---|---|
| `orbit` | `OrbitConfig(...)` | the plane every result uses (wp18-wp20, wp26, wp27); was 30.0, used nowhere |
| `grazing_km` | `100` | line of sight must clear Earth + this atmospheric margin |
| `max_range_km` | `None` | optional ISL link-range cap (None = pure line-of-sight) |

### Joint program

#### `sat7.joint.JointWeights`

Alpha / beta / gamma of the paper's eq 20.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `alpha` | `1` | weight on E_total (per joule) |
| `beta` | `1` | weight on D_tx    (per megabyte) |
| `gamma` | `1` | weight on T_total (per hour) |

#### `sat7.joint.JointConstraints`

The two floors of eq 20.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `a_min` | `0.9` | at least 90% of (weighted) objects must be reported at all |
| `i_min` | `0.5` | at least 50% of recoverable information must be preserved |

#### `sat7.joint.CostModel`

Bytes to energy and latency inside the joint program.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `p_tx_W` | `15` | ASSUMPTION (report 17): transmit power while keying |
| `r_tx_bps` | `2.53e+06` | SIM: effective downlink rate (orbit sim, report 17) |
| `p_proc_W` | `28` | ASSUMPTION (report 17): onboard processor power |
| `t_proc_s` | `{0: 0.0, 1: 0.0005, 2: 0.004, 3: 0.012}` |  |

### Dashboard

#### `demo_pipeline.Settings`

What the dashboard's sidebar sets for one run.

| field | default | meaning (the comment in the source) |
|---|---|---|
| `conf` | `0.25` | the onboard operating cut every experiment uses |
| `window` | `768` | SAHI window (detector-native) |
| `overlap` | `0.2` |  |
| `context` | `'prefilter'` | prefilter \| ships \| coast  (manual override, labelled) |
| `capture_h` | `2` | hours after the simulated mission epoch |
| `sweep` | `True` | latency vs capture time (24 extra link simulations) |

### Constants that are not dataclass fields

| name | value | meaning |
|---|---|---|
| `sat7.energy.DEFAULT_POWERS_W['cpu']` | `28` W | ASSUMPTION: no power was measured; swept in wp17 |
| `sat7.energy.DEFAULT_POWERS_W['gpu']` | `60` W | ASSUMPTION: no power was measured; swept in wp17 |
| `sat7.energy.DEFAULT_POWERS_W['tx']` | `15` W | ASSUMPTION: no power was measured; swept in wp17 |
| `sat7.energy.DEFAULT_R_TX_BPS` | `2.53e+06` bit/s | SIM: effective downlink rate from the orbit simulation |
| `sat7.orbit.LINK_PRESETS['cubesat_sband']` | 2 MHz, SNR 12 dB at zenith, cap 8 Mbps, efficiency 0.8 | link-budget preset (the default) |
| `sat7.orbit.LINK_PRESETS['smallsat_xband']` | 50 MHz, SNR 15 dB at zenith, cap 300 Mbps, efficiency 0.8 | link-budget preset |
| `sat7.scheduler.RAW_TILE_BYTES` | `1,769,472` B | one 768 x 768 tile as uncompressed 8-bit RGB: the raw baseline |
| `sat7.scheduler.COAST_TILE_BYTES_MEASURED` | `66,118` B | mean measured size of a coastal context tile (wp28) |

<!-- END GENERATED -->
