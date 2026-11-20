#!/usr/bin/env python3
"""Generated figures for the 19-Generative deck (run with a Python that has
matplotlib: /usr/local/bin/python3.12 figures.py). Writes SVGs to images/.

  attention-vs-ssm-cost.svg      per-sequence cost of attention (n^2) vs an
                                 SSM scan (n) as a function of packets per trace
  generative-landscape-timeline.svg  GAN -> diffusion -> transformer -> SSM,
                                 general-ML milestones above, networking below
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "Helvetica",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = Path(__file__).parent / "images"
GREY = "#555555"
BLUE = "#2b6cb0"
ORANGE = "#c05621"
GREEN = "#2f855a"
PURPLE = "#6b46c1"


def cost_figure():
    n = np.logspace(2, 6, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(n, n ** 2, color=ORANGE, lw=2.6, label="attention: $n^2$ pairwise scores")
    ax.loglog(n, n * np.log2(n), color=GREY, lw=1.6, ls="--", label="$n \\log n$ (for scale)")
    ax.loglog(n, n, color=GREEN, lw=2.6, label="SSM scan: $n$ state updates")
    for pkts, note in [(1e3, "one busy second"), (1e4, "one long TCP flow"), (1e5, "a multi-flow session")]:
        ax.axvline(pkts, color="#bbbbbb", lw=0.8, ls=":")
        ax.annotate(f"{int(pkts):,} packets\n{note}", (pkts, 1.6e2), ha="center", va="bottom",
                    fontsize=9.5, color=GREY)
        print(f"n={int(pkts):>7,}: attention {pkts**2:.0e} ops, ssm {pkts:.0e} ops, ratio {pkts:.0e}x")
    ax.set_xlabel("packets in the sequence, $n$")
    ax.set_ylabel("operations per sequence (log scale)")
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(1e2, 1e12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.grid(True, which="major", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(OUT / "attention-vs-ssm-cost.svg")


def timeline_figure():
    # (year, x for the label, tier 0..2, label, family, lane)  lane 1 = general ML, -1 = networking
    items = [
        (2014, 2014.0, 0, "GAN\nGoodfellow et al.", "gan", 1),
        (2017, 2017.0, 1, "Transformer\nVaswani et al.", "tf", 1),
        (2018, 2018.3, 0, "BERT", "tf", 1),
        (2020, 2020.0, 1, "DDPM\nHo et al.", "diff", 1),
        (2022, 2021.9, 0, "Stable Diffusion\nLoRA, ControlNet", "diff", 1),
        (2023, 2023.4, 2, "Mamba\nGu & Dao", "ssm", 1),
        (2020, 2020.0, 0, "DoppelGANger\nIMC", "gan", -1),
        (2021, 2021.0, 1, "nPrint\nCCS", "tf", -1),
        (2022, 2021.9, 0, "NetShare\nSIGCOMM", "gan", -1),
        (2022, 2022.6, 2, "ET-BERT\nWWW", "tf", -1),
        (2024, 2023.7, 1, "NetDiffusion\nSIGMETRICS", "diff", -1),
        (2024, 2024.6, 0, "NetLLM\nSIGCOMM", "tf", -1),
        (2025, 2025.3, 2, "NetSSM\nCoNEXT", "ssm", -1),
    ]
    colors = {"gan": ORANGE, "diff": BLUE, "tf": PURPLE, "ssm": GREEN}
    names = {"gan": "GANs", "diff": "Diffusion", "tf": "Transformers / LLMs", "ssm": "State space models"}
    tiers = [0.9, 1.85, 2.8]
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.axhline(0, color=GREY, lw=1.4)
    ax.set_xlim(2013.2, 2026.3)
    ax.set_ylim(-3.75, 3.75)
    for y in range(2014, 2026):
        ax.plot([y, y], [-0.12, 0.12], color=GREY, lw=1)
        ax.text(y, -0.4, str(y), ha="center", va="top", fontsize=13, color=GREY)
    for year, x, tier, label, fam, lane in items:
        yy = lane * tiers[tier]
        ax.plot([year, x], [0, yy], color=colors[fam], lw=1.2)
        ax.scatter([year], [0], s=70, color=colors[fam], zorder=3)
        ax.text(x, yy + lane * 0.1, label, ha="center", va="bottom" if lane > 0 else "top",
                fontsize=16, color=colors[fam],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[fam], lw=1.0))
    ax.text(2013.3, 3.55, "General machine learning", fontsize=14, color=GREY, va="center", style="italic")
    ax.text(2013.3, -3.55, "Applied to network traffic", fontsize=14, color=GREY, va="center", style="italic")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "generative-landscape-timeline.svg")


if __name__ == "__main__":
    cost_figure()
    timeline_figure()
    print("wrote", OUT / "attention-vs-ssm-cost.svg", OUT / "generative-landscape-timeline.svg")
