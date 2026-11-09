"""Generate the network-specific figures for the 12-Trees-Ensembles deck.

Writes SVG files into ../12-Trees-Ensembles/images/ and prints the impurity
arithmetic the slides quote, so the numbers on the slides are computed, not
typed. Usage:

    python3 src/trees_figures.py [--out DIR]
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RED, GREEN, BLUE, GREY = "#b91c1c", "#2e7d32", "#4a6fa5", "#555555"
REDBG, GREENBG = "#fde8e8", "#e3f1e3"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 15, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})

# Ten flows in (duration s, packets per second): the running example.
ATTACK = [(2, 120), (5, 90), (12, 150), (15, 110), (3, 30)]
BENIGN = [(20, 20), (40, 15), (7, 40), (6, 35), (25, 80)]

PNG_DIR = os.environ.get("TREES_FIG_PNG")

def save(fig, out, name):
    fig.tight_layout(); fig.savefig(os.path.join(out, name + ".svg"))
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True); fig.savefig(os.path.join(PNG_DIR, name + ".png"), dpi=110)
    plt.close(fig)

def gini(a, b):
    n = a + b
    if n == 0: return 0.0
    p = a / n
    return 1 - p * p - (1 - p) ** 2

def entropy(a, b):
    n = a + b
    if n == 0: return 0.0
    h = 0.0
    for c in (a, b):
        if c:
            p = c / n; h -= p * math.log2(p)
    return h

def split_counts(attack, benign, idx, thr, op):
    """Return (attack_left, benign_left, attack_right, benign_right) for feature idx."""
    test = (lambda v: v < thr) if op == "<" else (lambda v: v > thr)
    al = sum(test(p[idx]) for p in attack); bl = sum(test(p[idx]) for p in benign)
    return al, bl, len(attack) - al, len(benign) - bl

def weighted(fn, counts):
    al, bl, ar, br = counts
    n = al + bl + ar + br
    return ((al + bl) / n) * fn(al, bl) + ((ar + br) / n) * fn(ar, br)

def scatter(ax, attack, benign, xlabel="flow duration (s)", ylabel="packets per second"):
    ax.scatter(*zip(*attack), marker="x", s=110, c=RED, linewidths=2.5, label="attack", zorder=3)
    ax.scatter(*zip(*benign), marker="o", s=70, c=GREEN, label="benign", zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)

def fig_split_compare(out):
    """Two candidate root splits on the same ten flows, with child counts."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cases = [("duration < 10 s?", 0, 10, "<", "v"), ("packets per second > 50?", 1, 50, ">", "h")]
    results = {}
    for ax, (title, idx, thr, op, orient) in zip(axes, cases):
        scatter(ax, ATTACK, BENIGN)
        ax.set_xlim(0, 45); ax.set_ylim(0, 165)
        c = split_counts(ATTACK, BENIGN, idx, thr, op)
        g, h = weighted(gini, c), weighted(entropy, c)
        results[title] = (c, g, h)
        if orient == "v":
            ax.axvline(thr, color="#222", lw=2.5)
            ax.text(1, 152, f"{c[0]} attack, {c[1]} benign", fontsize=13, color=GREY)
            ax.text(22, 152, f"{c[2]} attack, {c[3]} benign", fontsize=13, color=GREY)
        else:
            ax.axhline(thr, color="#222", lw=2.5)
            ax.text(27, 140, f"{c[0]} attack, {c[1]} benign", fontsize=13, color=GREY)
            ax.text(27, 5, f"{c[2]} attack, {c[3]} benign", fontsize=13, color=GREY)
        ax.set_title(f"{title}\nweighted Gini = {g:.2f}", fontsize=15)
    axes[0].legend(loc="center right", frameon=False)
    save(fig, out, "split-compare")
    return results

def fig_entropy(out):
    """Left: the ten flows split on pps > 50 with per-child entropy. Right: H(p) curve."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; scatter(ax, ATTACK, BENIGN); ax.set_xlim(0, 45); ax.set_ylim(0, 165)
    ax.axhline(50, color="#222", lw=2.5)
    c = split_counts(ATTACK, BENIGN, 1, 50, ">")
    ha, hb = entropy(c[0], c[1]), entropy(c[2], c[3])
    ax.text(27, 140, f"{c[0]} attack, {c[1]} benign\nH = {ha:.2f} bits", fontsize=13, color=GREY)
    ax.text(20, 48, f"{c[2]} attack, {c[3]} benign\nH = {hb:.2f} bits", va="top", fontsize=13, color=GREY)
    ax.set_title(f"before the split: 5 and 5, H = {entropy(5,5):.2f} bit\nafter: weighted H = {weighted(entropy, c):.2f} bits", fontsize=15)
    ax.legend(loc="center right", frameon=False)
    ax = axes[1]
    p = np.linspace(0.001, 0.999, 400)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    ax.plot(p, H, color=BLUE, lw=2.5)
    for pp, lab, dx in [(0.5, "5 of 10 attack\nH = 1.00", 0.02), (0.8, "4 of 5 attack\nH = 0.72", 0.02), (1.0, "all one class\nH = 0", -0.02)]:
        hv = 0 if pp in (0, 1) else -pp*math.log2(pp)-(1-pp)*math.log2(1-pp)
        ax.plot([pp], [hv], "o", color=RED, ms=9, zorder=4)
        ax.annotate(lab, (pp, hv), xytext=(pp + dx, hv + 0.08 if pp < 1 else 0.14), fontsize=12,
                    ha="left" if pp < 1 else "right", color=GREY)
    ax.set_xlabel("fraction of flows in the region that are attacks, p"); ax.set_ylabel("entropy H(p), bits")
    ax.set_ylim(0, 1.25); ax.set_title("H(p) = −p log₂ p − (1−p) log₂ (1−p)", fontsize=15); ax.grid(alpha=0.3)
    save(fig, out, "entropy-example")
    return c, ha, hb

def fig_early_stopping(out):
    """An XOR-shaped flow population: no single split helps, two splits are perfect."""
    rng = np.random.default_rng(3)
    def cloud(cx, cy, n=7, sx=2.5, sy=14):
        return list(zip(rng.normal(cx, sx, n), rng.normal(cy, sy, n)))
    burst = cloud(5, 120); beacon = cloud(35, 25)      # attacks: short+fast, long+slow
    bulk = cloud(35, 120); interactive = cloud(5, 25)  # benign: long+fast, short+slow
    attack, benign = burst + beacon, bulk + interactive
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]; scatter(ax, attack, benign); ax.set_xlim(-3, 48); ax.set_ylim(-12, 180)
    ax.axvline(20, color="#222", lw=2.5); ax.axhline(70, color="#222", lw=2.5, ls="--")
    for x, y, t, col in [(5, 168, "flood", RED), (35, 168, "bulk transfer", GREEN), (5, -8, "interactive", GREEN), (35, -8, "slow beacon", RED)]:
        ax.text(x, y, t, ha="center", fontsize=13, color=col, fontweight="bold")
    ax.set_title("first split (solid): each child is still half attack, half benign\nsecond split (dashed): all four leaves are pure", fontsize=14)
    ax.text(46, 90, "× attack", color=RED, ha="right", fontsize=12); ax.text(46, 76, "● benign", color=GREEN, ha="right", fontsize=12)
    ax = axes[1]
    n = len(attack) + len(benign)
    depth = [0, 1, 2]; g = [gini(len(attack), len(benign)), gini(len(attack), len(benign)), 0.0]
    # depth-1 children are each 7 attack / 7 benign in this construction, so the weighted Gini is unchanged
    ax.plot(depth, g, "o-", color=BLUE, lw=2.5, ms=9)
    ax.set_xticks(depth); ax.set_xlabel("tree depth"); ax.set_ylabel("weighted Gini impurity of the leaves")
    ax.set_ylim(-0.05, 0.6); ax.grid(alpha=0.3)
    ax.annotate("early stopping quits here:\n\"the split did not help\"", (1, g[1]), xytext=(0.55, 0.22), fontsize=12, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_title("impurity by depth", fontsize=15)
    save(fig, out, "early-stopping")
    return g

def best_root(attack, benign):
    """Exhaustive CART root search over both features (midpoints between sorted values)."""
    best = None
    for idx, name in [(0, "duration"), (1, "packets per second")]:
        vals = sorted(set(p[idx] for p in attack + benign))
        for lo, hi in zip(vals, vals[1:]):
            thr = (lo + hi) / 2
            c = split_counts(attack, benign, idx, thr, "<")
            g = weighted(gini, c)
            if best is None or g < best[0] - 1e-12:
                best = (g, name, thr, c)
    return best

def fig_brittleness(out):
    """Same flows, three added: the root split changes feature."""
    attack = list(ATTACK); benign = list(BENIGN)
    extra = [(1, 20), (2, 25), (3, 12)]            # three short, slow attack flows (a probe)
    base = best_root(attack, benign)
    after = best_root(attack + extra, benign)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (att, title, br) in zip(axes, [(attack, "original training set", base), (attack + extra, "same set plus three new attack flows", after)]):
        scatter(ax, att, benign); ax.set_xlim(0, 45); ax.set_ylim(0, 165)
        if br[1] == "duration": ax.axvline(br[2], color="#222", lw=2.5)
        else: ax.axhline(br[2], color="#222", lw=2.5)
        ax.set_title(f"{title}\nroot split: {br[1]} {'<' if br[1]=='duration' else '>'} {br[2]:g}   (weighted Gini {br[0]:.2f})", fontsize=14)
    for x, y in extra:
        axes[1].add_patch(plt.Circle((x, y), 2.2, fill=False, color=RED, lw=1.5, ls="--"))
    axes[0].legend(loc="upper right", frameon=False)
    save(fig, out, "brittleness")
    return base, after

def fig_regression_partition(out):
    """Regression tree on flows: predict throughput from RTT and loss; three leaves with means."""
    rng = np.random.default_rng(7)
    pts = []
    for n, (rx, lx, mean, sd) in enumerate([((5, 48), (0.0, 0.9), 42, 4), ((5, 48), (1.1, 4.8), 9, 2), ((55, 195), (0.0, 4.8), 4, 1.2)]):
        for _ in range(7):
            pts.append((rng.uniform(*rx), rng.uniform(*lx), max(0.5, rng.normal(mean, sd))))
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.add_patch(Rectangle((0, 0), 50, 1, color="#dbeafe")); ax.add_patch(Rectangle((0, 1), 50, 4, color="#fef3c7")); ax.add_patch(Rectangle((50, 0), 150, 5, color="#fee2e2"))
    ax.plot([50, 50], [0, 5], color="#222", lw=2.5); ax.plot([0, 50], [1, 1], color="#222", lw=2.5)
    for r, l, t in pts:
        ax.plot(r, l, "o", color="#333", ms=5); ax.text(r + 2, l + 0.06, f"{t:.0f}", fontsize=10, color="#333")
    means = {}
    for name, sel in [("R1", lambda p: p[0] < 50 and p[1] < 1), ("R2", lambda p: p[0] < 50 and p[1] >= 1), ("R3", lambda p: p[0] >= 50)]:
        means[name] = np.mean([p[2] for p in pts if sel(p)])
    ax.text(48, 0.08, f"R1: mean {means['R1']:.0f} Mbps", ha="right", fontsize=13, fontweight="bold", color=BLUE)
    ax.text(25, 4.4, f"R2: mean {means['R2']:.0f} Mbps", ha="center", fontsize=13, fontweight="bold", color="#b45309")
    ax.text(125, 4.4, f"R3: mean {means['R3']:.0f} Mbps", ha="center", fontsize=13, fontweight="bold", color=RED)
    ax.set_xlim(0, 200); ax.set_ylim(0, 5); ax.set_xlabel("round-trip time (ms)"); ax.set_ylabel("packet loss (%)")
    ax.set_title("each point is a flow, labelled with its throughput (Mbps)", fontsize=13)
    save(fig, out, "regression-partition")
    return means

def fig_adaboost(out):
    """Two stumps for scan detection: ports first, then rate; circled points are the misses."""
    scans = [(60, 40), (80, 55), (120, 30), (200, 70), (90, 20)]
    slow = [(8, 0.5), (12, 0.8), (6, 0.3)]
    benign = [(3, 10), (5, 25), (2, 40), (8, 15), (10, 30), (4, 5), (4, 0.8)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, k in zip(axes, (1, 2)):
        ax.tick_params(labelsize=14)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.scatter(*zip(*(scans + slow)), marker="x", s=110, c=RED, linewidths=2.5, label="port scan", zorder=3)
        ax.scatter(*zip(*benign), marker="o", s=70, c=GREEN, label="benign", zorder=3)
        ax.set_xlim(1, 400); ax.set_ylim(0.1, 150); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("distinct destination ports", fontsize=17); ax.set_ylabel("packets per second", fontsize=17)
        if k == 1:
            ax.axvline(20, color="#222", lw=2.5)
            for x, y in slow: ax.add_patch(plt.Circle((x, y), 0.18, transform=ax.transData, fill=False, color="#222", lw=1.8))
            ax.set_title("stump 1: over 20 ports = scan\nmisses the three slow scans (circled)", fontsize=17)
            ax.legend(loc="lower right", frameon=False, fontsize=15)
        else:
            ax.axvline(20, color="#999", lw=1.5, ls="--"); ax.axhline(1, color="#222", lw=2.5)
            ax.add_patch(plt.Circle((4, 0.8), 0.18, transform=ax.transData, fill=False, color="#222", lw=1.8))
            ax.set_title("stump 2, reweighted: under 1 pkt/s = scan\ncatches the slow scans, misses one idle flow", fontsize=17)
    # log axes: draw circles as ellipses in data space instead
    for ax in axes:
        for p in ax.patches: p.remove()
    for x, y in slow:
        axes[0].plot([x], [y], "o", ms=22, mfc="none", mec="#222", mew=1.8, zorder=4)
    axes[1].plot([4], [0.8], "o", ms=22, mfc="none", mec="#222", mew=1.8, zorder=4)
    save(fig, out, "adaboost-scans")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "12-Trees-Ensembles", "images"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    r = fig_split_compare(a.out)
    for k, (c, g, h) in r.items():
        print(f"{k:28s} children {c}  Gini L {gini(c[0],c[1]):.3f} R {gini(c[2],c[3]):.3f} weighted {g:.3f} | entropy weighted {h:.3f}")
    print(f"parent: Gini {gini(5,5):.3f} entropy {entropy(5,5):.3f}")
    c, ha, hb = fig_entropy(a.out); print(f"entropy children {c}: {ha:.3f} {hb:.3f}")
    print("early stopping Gini by depth", [round(x, 3) for x in fig_early_stopping(a.out)])
    base, after = fig_brittleness(a.out); print("brittleness base", base, "\nbrittleness after", after)
    print("regression means", {k: round(v, 1) for k, v in fig_regression_partition(a.out).items()})
    fig_adaboost(a.out); print("wrote", sorted(os.listdir(a.out)))
