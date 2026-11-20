"""Generated figures for the 18-State-Space deck (matplotlib, SVG).

Run from the deck directory:  /usr/local/bin/python3.12 make-figures.py
Writes figures/*.svg. Every number plotted is one already quoted on the
slide it accompanies (NetSSM paper, Chu et al., CoNEXT 2025); the
multi-flow timeline and the tokenization example are illustrative.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.family": "Helvetica", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f5fa8", "#e0862a", "#1a8a4a", "#c8302e", "#888888", "#7b4ea3"
MAROON = "#800000"

# --- 1. attention (n^2) vs SSM (n) cost in sequence length
n = np.logspace(2, 6, 200)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.loglog(n, n**2, color=RED, lw=2.5, label="self-attention: $n^2$ pairwise scores")
ax.loglog(n, n, color=BLUE, lw=2.5, label="state-space model: $n$ state updates")
for x, lab in [(1e3, "1 K tokens"), (1e5, "100 K tokens")]:
    ax.axvline(x, color=GREY, lw=0.8, ls=":")
    ax.annotate(f"{lab}\nattention {x**2:.0e} · SSM {x:.0e}", (x, x**2), xytext=(8, -8),
                textcoords="offset points", fontsize=10.5, va="top")
ax.annotate("NetDiffusion-class\ntransformer context ≈ 12 K", (1.2e4, 1.2e4**2), xytext=(-150, 20),
            textcoords="offset points", fontsize=10.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.annotate("NetSSM training context 100 K", (1e5, 1e5), xytext=(-180, 25), textcoords="offset points",
            fontsize=10.5, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
ax.set_xlabel("sequence length $n$ (tokens)"); ax.set_ylabel("operations per layer (relative)")
ax.set_ylim(1e2, 1e13); ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.set_title("Cost per layer vs. sequence length, from the formulas", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "attention-vs-ssm-cost.svg"); plt.close(fig)
print("1K: attention", 1e6, "ssm", 1e3, "| 100K: attention", 1e10, "ssm", 1e5)

# --- 2. tokenization: packet bytes -> token sequence
fig, ax = plt.subplots(figsize=(12.4, 3.6)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 3.6); ax.axis("off")
# IPv4 header start of a Netflix-labelled packet (illustrative bytes)
hexes = ["45", "00", "05", "dc", "1a", "2b", "40", "00", "40", "06", "…"]
ids = [str(int(h, 16)) if h != "…" else "…" for h in hexes]
def box(x, y, w, h, text, fc, fs=12, ec="#555555", bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", family="Helvetica")
ax.text(0.1, 3.05, "packet bytes (hex), straight off the wire", fontsize=12, color="#444444")
ax.text(0.1, 1.05, "token sequence the model sees (one ID per byte, 0–255, plus special tokens)", fontsize=12, color="#444444")
x = 0.1; w = 0.62
for h in hexes:
    box(x, 2.2, w, 0.7, h, "#cfe3f5", fs=13); x += w + 0.06
box(x + 0.3, 2.2, 1.9, 0.7, "next packet…", "#eeeeee", fs=12)
x = 0.1
box(x, 0.2, 1.5, 0.7, "<|netflix|>", "#f2d9a8", fs=12, bold=True); x += 1.56
for i in ids:
    box(x, 0.2, w, 0.7, i, "#d9efd3", fs=13); x += w + 0.06
box(x + 0.02, 0.2, 1.0, 0.7, "<|pkt|>", "#f2d9a8", fs=12, bold=True)
box(x + 1.1, 0.2, 1.2, 0.7, "next packet…", "#eeeeee", fs=12)
for k in range(len(hexes)):
    xa = 0.1 + k * (w + 0.06) + w / 2
    ax.add_patch(FancyArrowPatch((xa, 2.15), (xa + 1.56, 0.95), arrowstyle="->", color="#999999",
                                 lw=0.9, mutation_scale=10))
ax.text(0.85, 1.6, "application label\nseeds generation", fontsize=10.5, color="#7a5a10", ha="center")
ax.text(x + 0.52, 1.6, "packet boundary", fontsize=10.5, color="#7a5a10", ha="center")
fig.savefig(OUT / "tokenization-bytes-to-tokens.svg", bbox_inches="tight"); plt.close(fig)

# --- 3. results: JSD and downstream accuracy, from the slide's tables
methods = ["NetShare", "NetDiffusion", "NetSSM"]; jsd = [0.16, 0.04, 0.02]; acc = [0.13, 0.16, 0.97]
cols = [GREY, ORANGE, MAROON]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, vals, title, ylab in [(a1, jsd, "Statistical similarity (JSD, lower is better)", "Jensen–Shannon divergence"),
                               (a2, acc, "Downstream ML utility (train on synthetic, test on real)", "random-forest accuracy")]:
    b = ax.bar(methods, vals, color=cols, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=12.5); ax.set_ylabel(ylab); ax.set_ylim(0, max(vals) * 1.18)
fig.tight_layout(); fig.savefig(OUT / "results-jsd-accuracy.svg"); plt.close(fig)
print("JSD", dict(zip(methods, jsd)), "acc", dict(zip(methods, acc)))

# --- 4. multi-flow session timeline (illustrative shape of a video-streaming session)
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(12.4, 3.4))
flows = [("CDN endpoint setup (TLS handshake, manifest)", BLUE, 0, [(0.0, 0.9, 8, 120)]),
         ("video segment download #1", ORANGE, 1, [(1.0, 2.6, 60, 1400)]),
         ("video segment download #2", GREEN, 2, [(2.9, 4.4, 60, 1400)]),
         ("control / telemetry (small, periodic)", PURPLE, 3, [(0.5, 5.0, 14, 90)])]
for name, c, row, segs in flows:
    for t0, t1, k, size in segs:
        ts = np.sort(rng.uniform(t0, t1, k)); sizes = rng.normal(size, size * 0.1, k).clip(40, 1500)
        ax.scatter(ts, np.full(k, row), s=sizes / 12, color=c, alpha=0.75, edgecolor="none")
        # ACKs from the client interleaved with the data
        ax.scatter(ts + 0.02, np.full(k, row) - 0.22, s=6, color=c, alpha=0.6, marker="|")
    ax.text(5.05, row, name, va="center", fontsize=11.5, color=c)
ax.set_yticks([]); ax.set_ylim(-0.7, 3.6); ax.set_xlim(0, 8.6); ax.spines["left"].set_visible(False)
ax.set_xlabel("time within one session (s)"); ax.set_title("One session, four interleaved flows (marker size scales with packet size; ticks are ACKs)", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT / "multiflow-session-timeline.svg"); plt.close(fig)

# --- 5. context and generation length: NetDiffusion vs NetSSM
fig, ax = plt.subplots(figsize=(7.2, 4.0))
labels = ["training context", "longest generated trace"]; nd = [12_000, 1_300]; ns = [100_000, 100_000]
xs = np.arange(2); wdt = 0.36
b1 = ax.bar(xs - wdt / 2, nd, wdt, color=ORANGE, label="NetDiffusion (transformer-class)")
b2 = ax.bar(xs + wdt / 2, ns, wdt, color=MAROON, label="NetSSM (Mamba-2)")
ax.set_yscale("log"); ax.set_ylim(500, 5e5); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=12.5)
ax.set_ylabel("tokens (log scale)"); ax.legend(frameon=False, fontsize=11, loc="upper left")
for r, v in list(zip(b1, nd)) + list(zip(b2, ns)):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v/1000:g} K", ha="center", fontsize=12, fontweight="bold")
ax.text(0, 2.2e5, "≈ 8×", ha="center", fontsize=14, color=MAROON, fontweight="bold")
ax.text(1, 2.2e5, "≈ 78×", ha="center", fontsize=14, color=MAROON, fontweight="bold")
ax.set_title("Context and generation length (≈ 100 tokens per packet)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "context-generation-length.svg"); plt.close(fig)
print("ratios", 100000 / 12000, 100000 / 1300)
