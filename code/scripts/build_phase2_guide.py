"""Build the Phase 2 & poster preparation guide (PDF).

    python scripts/build_phase2_guide.py

Output: ../report/IASTAM_P7_Phase2_Poster_Guide.pdf
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from build_report import BODY, CALLOUT, H1, H2, MUTED, P, SMALL, SPACE, bullets, figure, table, tag

TITLE = "Phase 2 and Poster — preparation guide"

# hex versions for matplotlib (the imported ones are ReportLab Color objects)
HX_SPACE, HX_OCEAN, HX_ACCENT, HX_GOOD, HX_MUTED = "#0B2545", "#1B6CA8", "#E07A1F", "#2E8B57", "#6B7280"


# --------------------------------------------------------------------------- figures


def poster_layout(out: Path) -> Path:
    """A0 portrait poster skeleton: what goes where, with size rules."""
    fig, ax = plt.subplots(figsize=(8.4, 11.9))
    ax.set_xlim(0, 84.1)
    ax.set_ylim(0, 118.9)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 82.9, 117.7, boxstyle="round,pad=0,rounding_size=1",
                                fc="white", ec=HX_SPACE, lw=2))

    def block(x, y, w, h, title, lines, col, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=col + "18", ec=col, lw=1.6))
        ax.text(x + 1.2, y + h - 2.0, title, fontsize=fs + 2.2, weight="bold", color=HX_SPACE, va="top")
        ax.text(x + 1.2, y + h - 4.6, lines, fontsize=fs, color="#1F2937", va="top", linespacing=1.5)

    # header
    ax.add_patch(FancyBboxPatch((3, 104), 78, 12, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=HX_SPACE, ec=HX_SPACE))
    ax.text(42, 111.5, "TRANSMITTING INFORMATION, NOT RAW DATA", fontsize=13, weight="bold",
            color="white", ha="center")
    ax.text(42, 108.4, "Onboard ship detection + confidence-aware level of detail + value-aware downlink",
            fontsize=8.5, color="#8DB8E0", ha="center")
    ax.text(42, 105.6, "Team <name>  •  IASTAM 6.0 • Track 4 • Problem 7  •  logos + QR to repo",
            fontsize=7.5, color="white", ha="center")

    x1, x2, x3, w = 3, 29.4, 55.8, 25.2
    block(x1, 78, w, 24, "1. THE PROBLEM",
          "• A satellite sees far more than it can send\n"
          "• REAL: 5 passes/day over Sfax,\n   34 min contact, 11.4 h gap\n"
          "• Most images: empty sea or cloud\n\n"
          "One big number, huge font:\n   “19% of tiles are duplicates”\n   or “93% of ships delivered”", HX_OCEAN)
    block(x1, 52, w, 24, "2. OUR IDEA IN 6 STEPS",
          "FIGURE: the 6-step diagram\n(simple_steps.png)\n\n"
          "Steps 4 and 5 highlighted:\n"
          "• how many bytes per ship\n"
          "• what to send first, across passes", HX_ACCENT)
    block(x1, 26, w, 24, "3. DATA WE USE",
          "• Airbus SPOT 1.5 m, 768 px tiles\n"
          "• 53,195 tiles • 81,723 ships\n"
          "• FIGURE: duplicate pair example\n\n"
          "• Leakage-free split:\n   1,617 groups would have leaked", HX_GOOD)
    block(x1, 3, w, 21, "8. WHAT'S NEXT",
          "• Detector metrics (WP1)\n• Calibration (WP2)\n"
          "• Measured level of detail (WP4)\n• Optimality gap (WP5)\n• Live demo", HX_MUTED)

    block(x2, 66, w, 36, "4. ARCHITECTURE",
          "BIG FIGURE: pipeline\n\n"
          "tiles → pre-filter → detector →\ncalibration → level of detail →\n"
          "scheduler → ground\n\n"
          "Keep boxes large, few words,\nour two blocks in orange", HX_OCEAN)
    block(x2, 40, w, 24, "5. LEVEL OF DETAIL",
          "TABLE, 5 rows max:\n"
          "sure + known → 40 B\n"
          "unsure → small chip\n"
          "dark vessel → progressive ROI\n"
          "coast/port → compressed tile\n"
          "very unsure → thumbnail\n\n"
          "Message: more bytes when less sure", HX_ACCENT)
    block(x2, 14, w, 24, "6. SCHEDULER",
          "max Σ vᵢ hᵢ(xᵢ)\n s.t. Σ bᵢ xᵢ ≤ C_pass\n\n"
          "• value per byte, with aging\n• progressive items can be cut\n"
          "• knapsack → greedy, gap measured\n\nOne small formula only!", HX_ACCENT)
    block(x2, 3, w, 9, "REFERENCES (small)",
          "6–8 key papers, 8 pt is fine here", HX_MUTED)

    block(x3, 70, w, 32, "7. RESULTS",
          "MAIN CHART (biggest figure):\nships delivered per strategy\n\n"
          "raw 0.4% • fixed patch 66.6%\nFIFO 68.5% • ours 93.0%\n\n"
          "Label it SIMULATION honestly", HX_GOOD)
    block(x3, 44, w, 24, "RESULTS (2)",
          "Load sweep chart:\nwhere each strategy breaks\n\n"
          "+ small table: latency\nmedian / 90th percentile", HX_GOOD)
    block(x3, 18, w, 24, "RELIABILITY",
          "• What if a ship is missed?\n   thumbnail + onboard buffer\n"
          "• Silent failure? 1% random\n   raw tiles audited\n"
          "• Detector crash → safe mode\n"
          "• No AIS = priority, not proof", HX_OCEAN)
    block(x3, 3, w, 13, "LEGEND + CONTACT",
          "REAL / SIM / LIT colour key\nQR code → repository + report\nteam e-mails", HX_MUTED)

    ax.text(42, 1.6, "A0 portrait (841 × 1189 mm) • title ≥ 72 pt • headings ≥ 36 pt • body ≥ 24 pt "
            "• figures ≥ 40% of the area • ≤ 800 words",
            fontsize=7.5, color=HX_ACCENT, ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def sahi_chart(out: Path) -> Path:
    w, t = 8000, 640
    ov = [0.0, 0.1, 0.2, 0.3]
    tiles = [(math.ceil((w - t) / int(t * (1 - o))) + 1) ** 2 for o in ov]
    base = tiles[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar([f"{int(o*100)}%" for o in ov] + ["edge-\ntriggered"],
                [n / base for n in tiles] + [(base + 2) / base],
                color=["#6B7280", "#6B7280", "#C0392B", "#C0392B", "#2E8B57"])
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_ylabel("onboard compute (x)")
    axes[0].set_title("Cost of overlap on one 8000 px frame", fontsize=10)
    for i, v in enumerate([n / base for n in tiles] + [(base + 2) / base]):
        axes[0].text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=8)
    sizes = np.array([20, 43, 100, 200, 380])
    axes[1].bar([str(s) for s in sizes], 100 * (1 - (1 - sizes / t) ** 2), color="#1B6CA8")
    axes[1].set_xlabel("ship size (px)")
    axes[1].set_ylabel("chance of straddling a seam (%)")
    axes[1].set_title("Why overlap matters: big ships are often cut", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- document


def on_page_guide(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(SPACE)
    canvas.rect(0, h - 1.1 * cm, w, 1.1 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial-Bold", 8.5)
    canvas.drawString(2 * cm, h - 0.72 * cm, "IASTAM 6.0 • Track 4 • Problem 7")
    canvas.setFont("Arial", 8.5)
    canvas.drawRightString(w - 2 * cm, h - 0.72 * cm, "Phase 2 and poster — preparation guide")
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial", 8)
    canvas.drawString(2 * cm, 1.1 * cm, f"Mid-review: 26 September 2026 • written {date.today():%d %B %Y}")
    canvas.drawRightString(w - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build(out: Path) -> None:
    story: list = []
    story += [P(TITLE, H1),
              P(f"IASTAM 6.0 • Track 4 • Problem 7 — {date.today():%d %B %Y}. "
                "Companion to the technical report: what to deliver for Phase 2, how to build the poster, "
                "and what to do about SAHI.", SMALL)]

    story += [P("1. Where we stand today", H2)]
    story += [table([
        ["Piece", "State", "Evidence"],
        ["Problem understanding + literature", "Done", "12 papers read, limitations verified in the text"],
        ["Architecture and design", "Done", "Pipeline, level of detail, scheduler formulation"],
        ["Code base + tests", "Done", "28 automated tests pass"],
        ["Contact windows " + tag("REAL"), "Done", "5 passes/day over Sfax, 11.4 h gap"],
        ["Scheduler comparison " + tag("SIM"), "Done", "68.5% → 93.0% from scheduling alone"],
        ["Data integrity (WP0) " + tag("REAL"), "Done", "53,195 tiles, 6,282 duplicate pairs, 0 leakage"],
        ["Detector (WP1)", "Not started", "Dataset is ready to train"],
        ["Calibration (WP2), measured LoD (WP4)", "Not started", "Phase 3"],
        ["Pitch video, interim paper", "Not started", "Due 26 September"],
    ], [5.4, 2.6, 9.0])]
    story += [P("<b>The gap that matters:</b> we have no detector numbers yet, and the jury's biggest block of "
                "points is “actual prototype progress”. Everything below is organised around closing that "
                "gap in three days.", CALLOUT)]

    story += [P("2. What Phase 2 actually requires", H2)]
    story += [table([
        ["Deliverable", "Rule (spec book)", "Our plan"],
        ["Pitch video", "Maximum 2 minutes: problem, core idea, technical approach",
         "1 take, screen recording + voice-over, no fancy editing"],
        ["Interim research paper", "Academic conventions, IEEE double column recommended; title/authors, "
         "abstract 150–250 words + 3–5 keywords, I Introduction, II Related Work, III Methodology, "
         "IV Experiments & Roadmap, V Conclusion, References",
         "Sections II and III are already written in the technical report"],
        ["Mid-review", "Scored out of 100; ≥ 60 required to continue", "See the scoring map below"],
        ["Poster session", "Open to all teams, even those not selected for the final pitch; templates and "
         "guidelines shared by the organisers after registration",
         "Build content now, drop it into the official template when it arrives"],
    ], [3.4, 7.3, 6.3])]

    story += [P("3. Three-day plan (23 → 26 September)", H2)]
    story += [table([
        ["When", "Task", "Owner", "Output"],
        ["Wed 23, evening", "Launch YOLOv8n training on Kaggle/Colab (dataset is ready)", "Detector",
         "Training running overnight"],
        ["Wed 23", "Paper skeleton in Overleaf, paste Related Work + Methodology from the report", "Paper",
         "Sections I–III drafted"],
        ["Thu 24", "Detector results: mAP50, recall by ship size, FP32 vs INT8, CPU latency", "Detector",
         "predictions.csv + table"],
        ["Thu 24", "Feed real detections into the scheduler; rerun the comparison", "You",
         "Real-data version of the main chart"],
        ["Thu 24", "Fair fixed-patch baseline (same coastal handling)", "Scheduler", "Honest comparison"],
        ["Fri 25", "Finish paper, insert figures, proofread; record the video", "Paper + all", "PDF + MP4"],
        ["Fri 25", "Poster content block by block (see section 5)", "Visuals", "Poster draft"],
        ["Sat 26", "Buffer, final checks, submit before the deadline", "All", "Submitted"],
    ], [2.8, 7.4, 2.2, 4.6])]
    story += [P("<b>Rule for these three days:</b> no new features. Only measurements, writing and rehearsal. "
                "A working small result beats an unfinished big one.", CALLOUT)]

    story += [PageBreak(), P("4. The 2-minute video and the paper", H2)]
    story += [table([
        ["Time", "What is on screen", "What you say"],
        ["0:00–0:20", "One satellite image, then the same image reduced to a few numbers",
         "The problem: a satellite sees far more than it can send; 5 passes a day, 34 minutes of contact"],
        ["0:20–0:50", "The 6-step diagram, steps 4–5 highlighted",
         "Our idea: send information about ships, and decide how many bytes each ship deserves"],
        ["0:50–1:20", "Architecture + level-of-detail table",
         "How it works: cheap filter, detector, confidence decides the detail level"],
        ["1:20–1:45", "The main result chart", "What we measured, and honestly what is simulated"],
        ["1:45–2:00", "Roadmap slide", "What comes before the final: real detector, calibration, demo"],
    ], [1.9, 6.2, 8.9])]
    story += bullets([
        "Record the screen with your voice; no music, no slow intro animation — the first 10 seconds decide attention.",
        "Say a number in the first 20 seconds (11.4 h between contacts).",
        "State clearly which results are simulated: juries trust teams that separate measured from expected.",
        "Keep a 10 s safety margin: a 2:05 video can be rejected.",
    ])
    story += [P("Paper: where each part already exists", H2)]
    story += [table([
        ["Section", "Source in our material", "Extra work"],
        ["Abstract + keywords", "Executive summary of the technical report", "Rewrite to 150–250 words"],
        ["I. Introduction", "Report sections 1–2 (bottleneck, real passes)", "Add our contributions as a list"],
        ["II. Related Work", "Report section 2 table with stated limitations", "Turn the table into prose"],
        ["III. Methodology", "Report section 3 + WP0 method", "Add the architecture figure"],
        ["IV. Experiments & Roadmap", "Report section 5 + detector results (Thursday)", "Add the WP plan table"],
        ["V. Conclusion", "Report section 7", "Two paragraphs"],
        ["References", "Report appendix C", "Convert to IEEE style"],
    ], [3.4, 8.0, 5.6])]
    story += [P("<b>Best Scientific Paper award:</b> what wins it is method honesty, not volume — the "
                "leakage-free split, the calibration story, confidence intervals, and the table of our own "
                "weaknesses. We already have all four.", CALLOUT)]

    story += [PageBreak(), P("5. The poster", H2)]
    story += [P("The organisers will send an official template; build the content now and drop it in. Standard "
                "format is A0 portrait (841 × 1189 mm). A poster is read from 1.5–2 m away, in a noisy "
                "room, in about 60 seconds.")]
    story += figure(poster_layout(RES / "poster_layout.png"), 12.0,
                    "Suggested layout: three columns, one message per block. Numbers in the blocks refer to the "
                    "reading order.")
    story += [P("Rules that decide whether it is read", H2)]
    story += [table([
        ["Do", "Don't"],
        ["One sentence as the main message, in the header", "A wall of paragraphs"],
        ["Figures ≥ 40% of the surface", "Screenshots of code or terminal output"],
        ["Body text ≥ 24 pt, headings ≥ 36 pt, title ≥ 72 pt", "Text below 20 pt — unreadable standing up"],
        ["Maximum ~800 words in total", "Full paper sections copied in"],
        ["Big numbers as headlines (93%, 11.4 h, 19%)", "Tables with 20 rows"],
        ["Mark clearly what is simulated", "Presenting targets as results"],
        ["QR code to the repository and the report", "URLs written out by hand"],
        ["Print 48 h early; bring a PDF backup on a USB key", "Printing the morning of the congress"],
    ], [8.5, 8.5])]
    story += [P("Prepare a <b>60-second spoken version</b> and a <b>5-minute version</b>. Most visitors get the "
                "first; the jury gets the second.", CALLOUT)]

    story += [Spacer(1, 6), P("6. SAHI: should we use it?", H2)]
    story += [P("SAHI (Slicing Aided Hyper Inference) cuts a large frame into overlapping tiles so an object on a "
                "tile border is not split in half, then merges the boxes. The overlap is what costs us.")]
    story += figure(sahi_chart(RES / "sahi_cost.png"), 16.0,
                    "Left: onboard compute for one 8000 px frame with 640 px tiles. Right: probability that a ship "
                    "straddles a tile border, by ship size (Airbus boxes reach 380 px).")
    story += [table([
        ["Finding", "Number", "Consequence"],
        ["Standard 20% overlap", "169 → 256 tiles = 1.51× compute", "Expensive where energy is the constraint"],
        ["Median ship (43 px)", "13% chance of being cut", "Moderate problem"],
        ["Large ships (200–380 px)", "53–84% chance of being cut", "Serious: the most valuable targets"],
        ["Our data today", "Airbus tiles are already 768 px", "SAHI changes nothing for the current results"],
    ], [4.6, 5.6, 6.8])]
    story += [P("Our alternative: edge-triggered re-inference", H2)]
    story += [P("Instead of paying 51% more compute everywhere, spend it only where a ship might be cut. The cheap "
                "classic pre-filter already finds bright blobs; if a blob or a detection <i>touches a tile border</i>, "
                "run one extra inference on a window centred on that seam. Cost measured on a typical frame: "
                "about <b>1.01×</b> instead of 1.51×.")]
    story += bullets([
        "It reuses WP3 (the classic filter) to decide where WP1 (the detector) must look again — the two parts "
        "of our system cooperate.",
        "It is measurable: recall of ships near seams, and compute cost, for three policies (no overlap, SAHI 20%, "
        "edge-triggered).",
        "It is an honest comparison against a well-known baseline (SAHI raised AP by 5–7 points on aerial "
        "datasets), which is exactly what a jury wants to see.",
    ])
    story += [P("<b>Timing: this is Phase 3 work, not before the mid-review.</b> Mention it in the paper's roadmap "
                "as planned work, with the cost numbers above — they already show we understand the trade-off.",
                CALLOUT)]
    story += [P("Experiment protocol (Phase 3)", H2)]
    story += [table([
        ["Step", "How"],
        ["Build the test", "Take Airbus tiles and slice each into 2×2 sub-tiles: ships on the seams are known "
                           "from the labels"],
        ["Policies", "(a) no overlap, (b) SAHI 20% overlap + box merging, (c) edge-triggered re-inference"],
        ["Measure", "Recall of seam ships, recall overall, number of inferences, runtime"],
        ["Report", "One chart: recall versus compute cost for the three policies"],
    ], [3.4, 13.6])]

    story += [PageBreak(), P("7. Scoring map: where the 100 points come from", H2)]
    story += [table([
        ["Criterion", "Pts", "What we show", "Status"],
        ["Understanding of the problem", "15", "Real contact windows, verified literature limitations", "Strong"],
        ["Relevance and originality", "15", "Confidence-aware level of detail + multi-pass scheduler", "Strong"],
        ["Technical architecture", "20", "Pipeline, interfaces, design patterns", "Strong"],
        ["Feasibility and realism", "15", "Realistic link, COTS evidence, FMEA", "Strong"],
        ["Actual prototype progress", "20", "Code, 28 tests, WP0 on real data, detector (this week)", "Needs WP1"],
        ["Methodology and planning", "10", "Leakage-free split, baselines, confidence intervals", "Strong"],
        ["Presentation", "5", "Video, poster, rehearsed answers", "To do"],
    ], [5.0, 1.2, 7.4, 3.4])]
    story += [P("8. Questions to rehearse", H2)]
    story += bullets([
        "Which of your results are measured and which are simulated?",
        "What happens when the detector misses a ship?",
        "Why greedy scheduling, and how far from optimal is it?",
        "How do you know your test scores are not inflated by duplicate images?",
        "Why not simply compress the images better?",
        "What does “no AIS signal” actually prove?",
        "Is your comparison against Φ-sat-2 fair?",
        "What would you do differently with a real satellite?",
    ])
    story += [Spacer(1, 8), P("Every one of these has an answer in the technical report; make sure each team member "
                              "can give it in 30 seconds.", CALLOUT)]

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm,
                            bottomMargin=1.8 * cm, title="IASTAM 6.0 Problem 7 - Phase 2 and poster guide",
                            author="Team")
    doc.build(story, onFirstPage=on_page_guide, onLaterPages=on_page_guide)


if __name__ == "__main__":
    target = PROJECT / "paper" / "IASTAM_P7_Phase2_Poster_Guide.pdf"
    build(target)
    print(target)
