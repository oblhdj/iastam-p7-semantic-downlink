"""Self-contained project report: explains everything from zero, ready to present.

    python scripts/build_overview.py   ->  ../report/IASTAM_P7_Project_Overview.pdf
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from build_report import (ACCENT, CALLOUT, H1, H2, MUTED, P, SMALL, SPACE, bullets, duplicate_grid, figure,
                          load, pct, table)


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(SPACE)
    canvas.rect(0, h - 1.1 * cm, w, 1.1 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial-Bold", 8.5)
    canvas.drawString(2 * cm, h - 0.72 * cm, "Transmitting Information, Not Raw Data")
    canvas.setFont("Arial", 8.5)
    canvas.drawRightString(w - 2 * cm, h - 0.72 * cm, "IASTAM 6.0 • Track 4 • Problem 7")
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial", 8)
    canvas.drawString(2 * cm, 1.1 * cm, f"Project report • {date.today():%d %B %Y}")
    canvas.drawRightString(w - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(SPACE)
    canvas.rect(0, h * 0.58, w, h * 0.42, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h * 0.58 - 0.22 * cm, w, 0.22 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial", 11)
    canvas.drawString(2.2 * cm, h - 3.0 * cm, "IASTAM 6.0 TECHNICAL CHALLENGE • TRACK 4 • PROBLEM 7")
    canvas.setFont("Arial-Bold", 29)
    canvas.drawString(2.2 * cm, h - 5.4 * cm, "Transmitting Information,")
    canvas.drawString(2.2 * cm, h - 6.7 * cm, "Not Raw Data")
    canvas.setFont("Arial", 13)
    canvas.drawString(2.2 * cm, h - 8.3 * cm, "A satellite that sends what matters about ships,")
    canvas.drawString(2.2 * cm, h - 9.0 * cm, "instead of sending pictures of empty sea")
    canvas.setFillColor(SPACE)
    canvas.setFont("Arial", 11)
    y = h * 0.58 - 2.0 * cm
    for line in ("Project report", "Team: <Your Team Name> — Computer Engineering, 2nd year",
                 f"{date.today():%d %B %Y}"):
        canvas.drawString(2.2 * cm, y, line)
        y -= 0.7 * cm
    canvas.restoreState()


def build(out: Path) -> None:
    d = load()
    sim, passes, wp0 = d["sim"], d["passes"], d["wp0"]

    def row(name):
        return sim[sim.strategy == name].iloc[0] if sim is not None and (sim.strategy == name).any() else None

    greedy, fifo = row("Ours: LoD + value-greedy"), row("Ours: LoD + FIFO")
    patch, raw = row("Phi-sat-2 style (patch, FIFO)"), row("Bent pipe (raw, FIFO)")

    story: list = [PageBreak()]

    # ---------------------------------------------------------------- 1
    story += [P("1. What we were asked to solve", H1)]
    story += [P("The IASTAM 6.0 challenge asks teams to work on the future of computing in space. We chose "
                "<b>Problem 7</b>, in the networks and communications track. Its statement is short:")]
    story += [P("<i>“A satellite can generate a large volume of images while only a small part is genuinely "
                "useful to the mission. How can the volume to be transmitted be strongly reduced while preserving "
                "the information necessary for decision-making on the ground?”</i> The example given is a "
                "mission looking for <b>ships</b> at sea: should the satellite send all the images, or only the "
                "detections and the zones of interest?", CALLOUT)]
    story += [P("The jury evaluates four things: how much the data volume is reduced, whether the useful "
                "information survives, how accurate the results are, and how long the ground waits for them.")]

    story += [P("2. Why this is a real problem", H2)]
    story += bullets([
        "A satellite can only send data while it passes over a ground station. The rest of the time the radio is "
        "useless, and the images pile up onboard.",
        "Most of what a camera records at sea is worthless: empty water, or clouds hiding everything.",
        "Sensors keep improving, so satellites produce more data every year, while the radio link does not improve "
        "at the same speed.",
    ])
    if passes is not None:
        story += [P("We did not take these statements on trust. We computed the real orbit of a satellite at 500 km "
                    "above the Earth and the visibility of a ground station in <b>Sfax, Tunisia</b>:")]
        story += [table([
            ["What we computed", "Result"],
            ["Passes over the station per day", f"{len(passes)}"],
            ["Total contact time per day", f"{passes.duration_min.sum():.1f} minutes"],
            ["Longest wait between two contacts", "11.4 hours"],
            ["Data one pass can carry (small satellite radio)", f"{passes.capacity_MB.mean():.0f} MB on average"],
        ], [9.0, 8.0])]
        story += [P("So the satellite is alone with its data for hours, and when the radio finally works it has a "
                    "few minutes. Sending raw pictures is hopeless: a single large image can fill an entire pass.")]

    # ---------------------------------------------------------------- 3
    story += [P("3. What is done today", H2)]
    story += [table([
        ["Approach", "What it does", "Its limit"],
        ["Send everything (“bent pipe”)", "The satellite is a simple relay", "The link saturates; most of "
         "what arrives is empty sea"],
        ["Compress the images", "Standard image compression", "A smaller picture of empty water is still a picture "
         "of empty water"],
        ["Throw away bad images (ESA Φ-sat-1, 2020)", "A neural network onboard rejects cloudy images",
         "Only decides keep or discard"],
        ["Send detections (ESA Φ-sat-2, 2024)", "Detects ships onboard, sends a small patch and a position",
         "Sends the same amount for every ship, whether obvious or doubtful"],
    ], [4.2, 6.4, 6.4])]
    story += [P("Nobody, in what we read, decides <b>how much detail each detection deserves</b> and <b>in which "
                "order to send things</b> when several hours of data are waiting for a few minutes of radio. "
                "That is the gap we work on.")]

    # ---------------------------------------------------------------- 4
    story += [PageBreak(), P("4. Our solution", H1)]
    story += [P("The satellite stops behaving like a camera that ships pictures, and behaves like an observer that "
                "reports what it sees. Two decisions are ours: <b>how many bytes each ship deserves</b>, and "
                "<b>what to send first</b>.")]
    story += figure(PROJECT / "simple_steps.png", 16.5, "The system in six steps. Steps 4 and 5, in orange, are our "
                    "contribution; the rest is standard practice.")
    story += [P("Decision 1 — how much detail per ship", H2)]
    story += [P("A detector gives each ship a confidence score. Most systems send less when they are unsure. "
                "We do the opposite: <b>the less certain the satellite is, the more evidence it sends</b>.")]
    story += [table([
        ["Situation", "What is sent", "Size"],
        ["Certain, and the ship already announces itself by radio (AIS)", "Just the facts: position, size, heading",
         "about 40 bytes"],
        ["Uncertain", "A small image of the ship so the ground can confirm", "1–3 KB"],
        ["A ship with no radio identification (“dark vessel”)", "A larger image that can be cut short if "
         "the radio time runs out", "10–50 KB"],
        ["A port, or a coastline full of boats", "The compressed area itself, because detectors are unreliable there",
         "10–80 KB"],
        ["Nothing convincing found", "A tiny preview of the whole tile, as a safety net", "about 1 KB"],
    ], [6.2, 7.4, 3.4])]
    story += [P("Decision 2 — what to send first", H2)]
    story += [P("Everything waiting onboard is queued with a value and a size. When a pass starts, the satellite "
                "sends the items with the best <b>value per byte</b> first, items that have waited a long time gain "
                "priority, and large images can be cut short when the pass ends. This is the classic knapsack "
                "problem: fill a limited bag with the most valuable objects.")]
    story += [P("A missed ship is never silently lost: a preview of every tile is sent, the raw data stays onboard "
                "for a while so the ground can ask for it again, and the ground checks detections against public "
                "ship-tracking data.", CALLOUT)]

    # ---------------------------------------------------------------- 5
    story += [PageBreak(), P("5. What we built and what it shows", H1)]
    story += [P("Everything below runs today as a Python program, with 28 automatic tests. Two kinds of results "
                "must not be confused: measurements on <b>real satellite data</b>, and a <b>simulation</b> of a day "
                "in orbit where the ships are generated by us.")]

    if wp0:
        story += [P("Real data: cleaning the ship dataset", H2)]
        story += [P("We use the Airbus Ship Detection dataset: real satellite images at 1.5 m per pixel, cut into "
                    f"tiles of 768 × 768 pixels. We kept {wp0['selected_tiles']:,} tiles containing "
                    f"{wp0['ships']:,} ships.")]
        story += [P("These tiles are cut out of larger scenes, and neighbouring cuts overlap, so <b>the same ship "
                    "can appear in two different images</b>. If one copy trains the detector and the other tests it, "
                    "the test is meaningless. We therefore built a tool that proves when two tiles show the same "
                    "place, and keeps such tiles on the same side of the split.")]
        story += [table([
            ["Measured on the real dataset", "Result"],
            ["Tiles analysed", f"{wp0['selected_tiles']:,}"],
            ["Pairs of tiles proven to show the same place", f"{wp0['duplicate_pairs']:,}"],
            ["Tiles having at least one duplicate", f"{wp0['tiles_in_duplicate_groups']:,} "
                                                    f"({pct(wp0['tiles_in_duplicate_groups'] / wp0['selected_tiles'])})"],
            ["Groups a naive random split would have leaked", f"{wp0['groups_a_naive_split_would_leak']:,}"],
            ["Groups leaking in our split", f"{wp0['groups_spanning_splits']}"],
        ], [10.0, 7.0])]
        grid = duplicate_grid(RES / "wp0_duplicates_grid.jpg")
        if grid:
            story += figure(grid, 16.0, "Real examples found by the tool: the same place photographed with a shifted "
                            "frame. Nearly one tile in five has such a twin.")

    if sim is not None and greedy is not None:
        story += [P("Simulation: one day in orbit", H2)]
        story += [P("We simulate a full day: the real passes over Sfax, a small-satellite radio, and a stream of "
                    "images with ships. We then compare what reaches the ground under different strategies.")]
        story += [table([
            ["Strategy", "Ships that reach the ground", "Of which unidentified ships"],
            ["Send raw images", pct(raw.ship_recall), pct(raw.dark_recall)],
            ["Send a fixed patch per ship (Φ-sat-2 style)", pct(patch.ship_recall), pct(patch.dark_recall)],
            ["Our detail levels, oldest sent first", pct(fifo.ship_recall), pct(fifo.dark_recall)],
            ["<b>Our detail levels + our priority order</b>", f"<b>{pct(greedy.ship_recall)}</b>",
             f"<b>{pct(greedy.dark_recall)}</b>"],
        ], [7.6, 5.0, 4.4])]
        reduction = raw.MB_offered / greedy.MB_offered if greedy.MB_offered else float("nan")
        story += [P(f"Our satellite offers <b>{greedy.MB_offered:.0f} MB</b> instead of "
                    f"{raw.MB_offered / 1000:.0f} GB of raw images — about <b>{reduction:.0f} times less "
                    f"data</b> — and still reports {pct(greedy.ship_recall)} of the ships. The payload sizes "
                    "used here were measured on real Airbus ship images, not guessed.", CALLOUT)]
        story += [P("At this imaging rate everything fits in the available radio time, so the sending order changes "
                    "the <b>waiting time</b> rather than the number of ships: an unidentified ship arrives after "
                    f"{greedy.dark_latency_med_h:.1f} h instead of {fifo.dark_latency_med_h:.1f} h. The order "
                    "becomes decisive when the satellite images more: at twice this rate the simple order delivers "
                    "69% of ships and ours 97%; at four times, 34% against 83%.")]
        story += figure(RES / "sim_recall.png", 15.0, "Simulation: ships reaching the ground in one day. Plain bars: "
                        "all ships. Hatched bars: ships without radio identification.")

    # ---------------------------------------------------------------- 6
    story += [PageBreak(), P("6. What is not finished, and what is next", H1)]
    story += [table([
        ["Question", "Where we stand"],
        ["Does the detector work on real images?", "The dataset is clean and ready; training is the next step"],
        ["Is the confidence score trustworthy?", "Detectors are usually over-confident; we will correct it and "
                                                 "publish the check"],
        ["Are the byte sizes realistic?", "Today they are estimates; we will measure them on real ship images"],
        ["Is the sending order the best possible?", "We will compare it with the exact optimum on small cases"],
        ["What about ships cut between two tiles?", "Large ships are cut 53–84% of the time; we propose to "
                                                   "re-examine only the borders, at about 1% extra computing"],
        ["Does it work on other satellites?", "Planned test on free Sentinel-2 images of the Tunisian coast"],
    ], [7.0, 10.0])]
    story += [P("Honesty rule we follow", H2)]
    story += bullets([
        "Every figure says whether it is a measurement on real data, a simulation, or a number from a published paper.",
        "We list the weaknesses of our own work, including a comparison that currently favours us unfairly.",
        "Three versions of our duplicate-finding tool were thrown away after we looked at the evidence and saw it "
        "was wrong: unrelated images of empty sea, of fog, and of scene edges were being matched.",
    ])

    story += [P("7. How to present this in three minutes", H1)]
    story += [table([
        ["Time", "What to say"],
        ["0:00–0:30", "A satellite sees far more than it can send: five passes a day over Sfax, 34 minutes of "
                           "radio, up to eleven hours of silence. And most images show empty sea."],
        ["0:30–1:00", "So we stop sending pictures. The satellite finds the ships and reports them. Show the "
                           "six-step figure."],
        ["1:00–1:45", "Our first idea: the less sure the satellite is, the more evidence it sends. A certain, "
                           "identified ship costs forty bytes; a suspicious one gets a real image."],
        ["1:45–2:30", "Our second idea: when the radio opens, send the most valuable bytes first. In our "
                           "simulation this delivers 97% of the ships with about 440 times less data than raw "
                           "images, and suspicious ships arrive an hour earlier — much more when the satellite "
                           "images heavily."],
        ["2:30–3:00", "What is real and what is not: the orbit and the dataset cleaning are measured, the day "
                           "in orbit is simulated, and the detector is our next step."],
    ], [2.4, 14.6])]
    story += [Spacer(1, 6), P("Expect these questions: what if a ship is missed (preview + data kept onboard + the "
                              "ground can ask again), how do you know your test is not inflated (we proved and "
                              "removed the duplicates), and does “no radio identification” prove "
                              "wrongdoing (no — it raises priority, a human decides).", CALLOUT)]

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm,
                            bottomMargin=1.8 * cm, title="Transmitting Information, Not Raw Data - project report",
                            author="Team")
    doc.build(story, onFirstPage=on_cover, onLaterPages=on_page)


if __name__ == "__main__":
    target = PROJECT / "report" / "IASTAM_P7_Project_Overview.pdf"
    build(target)
    print(target)
