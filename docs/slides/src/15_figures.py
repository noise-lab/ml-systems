"""Generated figures for the 15-Dimensionality-Reduction deck (matplotlib, SVG).

Run from docs/slides:  python3 src/15_figures.py
Writes to 15-Dimensionality-Reduction/images/gen-*.svg and prints the numbers
the slides quote.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

plt.rcParams.update({"svg.fonttype": "none", "font.family": "Helvetica", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = Path(__file__).resolve().parent.parent / "15-Dimensionality-Reduction" / "images"
BLUE, ORANGE, GREY, RED = "#1f5fa8", "#e0862a", "#777777", "#c8302e"
rng = np.random.default_rng(3)

# --- 1. PCA in pictures: correlated 2-D cloud, PC arrows, projection onto PC1
n = 300
X = rng.multivariate_normal([0, 0], [[3.0, 2.2], [2.2, 2.0]], n)
pca = PCA(2).fit(X); V = pca.components_; lam = pca.explained_variance_
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.55)
for k, (vec, var, col, lab) in enumerate(zip(V, lam, [RED, ORANGE], ["PC1", "PC2"])):
    L = 2.2 * np.sqrt(var)
    ax.annotate("", xy=vec * L, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=3))
    ax.text(*(vec * L * 1.12), lab, color=col, fontsize=14, fontweight="bold", ha="center")
ax.set_title("Original space: the two principal axes"); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
ax = axes[1]
Z1 = X @ V[0]; P = np.outer(Z1, V[0])
ax.scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.25)
for a, b in zip(X[::6], P[::6]): ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY, lw=0.6)
ax.scatter(P[:, 0], P[:, 1], s=16, color=ORANGE, zorder=3)
ax.set_title("Projected onto PC1: the 1-D shadow"); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
fig.tight_layout(); fig.savefig(OUT / "gen-pca-arrows-projection.svg"); plt.close(fig)
print("PC1 explains", round(pca.explained_variance_ratio_[0] * 100, 1), "% of the variance")

# --- 2. Rotation into the PC basis
Z = pca.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
axes[0].scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.55)
for vec, var, col in zip(V, lam, [RED, ORANGE]):
    L = 2.2 * np.sqrt(var); axes[0].annotate("", xy=vec * L, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=3))
axes[0].set_title("Input axes"); axes[0].set_xlabel("feature 1"); axes[0].set_ylabel("feature 2")
axes[1].scatter(Z[:, 0], Z[:, 1], s=14, color=BLUE, alpha=0.55)
axes[1].annotate("", xy=(2.2 * np.sqrt(lam[0]), 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=3))
axes[1].annotate("", xy=(0, 2.2 * np.sqrt(lam[1])), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=3))
axes[1].set_title("Principal-component axes (same points, rotated)"); axes[1].set_xlabel("PC1 score"); axes[1].set_ylabel("PC2 score")
for ax in axes: ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
fig.tight_layout(); fig.savefig(OUT / "gen-pca-rotation.svg"); plt.close(fig)

# --- 3. Scree plot on a 20-feature synthetic flow-feature matrix with decaying structure
p = 20; latent = rng.normal(size=(600, 4)) @ rng.normal(size=(4, p)) * np.array([3, 2, 1.2, 0.8])[:, None].sum(0) / 4
Xs = latent + rng.normal(scale=0.6, size=(600, p))
ev = PCA().fit(Xs).explained_variance_ratio_; cum = np.cumsum(ev)
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=100)
ax.bar(np.arange(1, p + 1), ev * 100, color="#cfe3f5", edgecolor=BLUE, label="variance explained by each PC")
ax.plot(np.arange(1, p + 1), cum * 100, "o-", color=RED, lw=2, label="cumulative")
k90 = int(np.argmax(cum >= 0.9)) + 1
ax.axhline(90, color=GREY, ls="--", lw=1); ax.text(p, 91, "90%", color=GREY, ha="right", fontsize=11)
ax.axvline(k90, color=GREY, ls=":", lw=1); ax.text(k90 + 0.2, 20, f"elbow: {k90} PCs\nreach 90%", color=GREY, fontsize=11)
ax.set_xlabel("number of principal components"); ax.set_ylabel("% of variance"); ax.set_xticks(range(1, p + 1, 1))
ax.set_title("Scree plot: 20 flow features, four of them doing the work"); ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "gen-scree.svg"); plt.close(fig)
print("scree: first 4 PCs explain", round(cum[3] * 100, 1), "%; 90% reached at", k90)

# --- 4. PCA for anomaly detection: a normal cloud along PC1 and one point far along PC2
rng2 = np.random.default_rng(11)
n = 260
Xn = rng2.multivariate_normal([0, 0], [[3.0, 2.2], [2.2, 2.0]], n)
pca2 = PCA(2).fit(Xn); V2 = pca2.components_
if V2[0] @ np.array([1, 1]) < 0: V2[0] = -V2[0]
if V2[1] @ np.array([-1, 1]) < 0: V2[1] = -V2[1]
anom = np.array([-1.2, 3.6])                       # few bytes, many packets: off the normal line
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
ax = axes[0]
ax.scatter(Xn[:, 0], Xn[:, 1], s=14, color=BLUE, alpha=0.5, label="normal traffic")
for vec, col, lab, L in zip(V2, [RED, ORANGE], ["PC1", "PC2"], [5.0, 2.2]):
    ax.annotate("", xy=vec * L, xytext=-vec * L, arrowprops=dict(arrowstyle="<->", color=col, lw=2.5))
    ax.text(*(vec * L * 1.12), lab, color=col, fontsize=14, fontweight="bold", ha="center")
ax.scatter(*anom, s=140, marker="*", color=RED, zorder=6, label="anomaly")
ax.annotate("many packets,\nfew bytes", anom, xytext=(-5.6, 3.9), fontsize=12, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.set_xlabel("bytes per second"); ax.set_ylabel("packets per second"); ax.set_title("Normal flows lie along PC1")
ax.set_aspect("equal"); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-5, 5); ax.legend(loc="lower right", fontsize=11, frameon=False)
ax = axes[1]
Zn = Xn @ V2.T; za = anom @ V2.T
ax.scatter(Zn[:, 0], Zn[:, 1], s=14, color=BLUE, alpha=0.5)
ax.scatter(*za, s=140, marker="*", color=RED, zorder=6)
ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
ax.annotate(f"PC2 score {za[1]:.1f}: far from every\nnormal point in the residual direction", za, xytext=(-6.2, 3.9), fontsize=12, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.set_xlabel("PC1 score (the normal relationship)"); ax.set_ylabel("PC2 score (the residual)"); ax.set_title("In PC coordinates the anomaly is one number")
ax.set_aspect("equal"); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-5, 5)
fig.tight_layout(); fig.savefig(OUT / "gen-pca-anomaly.svg"); plt.close(fig)

# --- 5. autoencoder drawn neuron by neuron, as on the board: x (100) squeezed to 10 and rebuilt as x'
from matplotlib.patches import FancyBboxPatch
sizes = [6, 5, 4, 3, 4, 5, 6]
names = ["x\n100 features", "64", "32", "bottleneck\n10", "32", "64", "x′\n100 features"]
fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=100)
xs = np.linspace(0, 12, len(sizes)); ys = {}
for li, (n_, xpos) in enumerate(zip(sizes, xs)):
    yy = np.linspace(-(n_ - 1) / 2, (n_ - 1) / 2, n_) * 0.75; ys[li] = yy
    col = "#cfe3f5" if li == 0 else ("#d9efd3" if li == len(sizes) - 1 else ("#f2d9a8" if li == 3 else "#e8f0f8"))
    for y in yy: ax.add_patch(plt.Circle((xpos, y), 0.22, facecolor=col, edgecolor="#555555", lw=1.2, zorder=4))
    if li in (0, len(sizes) - 1): ax.text(xpos, yy[0] - 0.55, "⋮", ha="center", va="top", fontsize=18, color="#555555")
    ax.text(xpos, yy[-1] + 0.6, names[li], ha="center", va="bottom", fontsize=13, fontweight="bold" if li in (0, 3, 6) else "normal")
for li in range(len(sizes) - 1):
    for y1 in ys[li]:
        for y2 in ys[li + 1]: ax.plot([xs[li], xs[li + 1]], [y1, y2], color="#999999", lw=0.5, zorder=1)
ax.add_patch(FancyBboxPatch((xs[3] - 0.5, ys[3][0] - 0.5), 1.0, ys[3][-1] - ys[3][0] + 1.0, boxstyle="round,pad=0.05", fill=False, edgecolor=ORANGE, lw=2.2, zorder=5))
ax.text(xs[1], -3.0, "encoder", ha="center", fontsize=13, color="#555555"); ax.text(xs[5], -3.0, "decoder", ha="center", fontsize=13, color="#555555")
ax.annotate("", xy=(xs[2] + 0.4, -2.75), xytext=(xs[0] - 0.3, -2.75), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
ax.annotate("", xy=(xs[6] + 0.3, -2.75), xytext=(xs[4] - 0.4, -2.75), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
ax.add_patch(FancyBboxPatch((3.6, 3.35), 4.8, 0.95, boxstyle="round,pad=0.08", facecolor="#fdeeea", edgecolor=RED, lw=2, zorder=5))
ax.text(6.0, 3.82, "reconstruction loss  ‖x − x′‖²", ha="center", va="center", fontsize=14, color=RED, zorder=6)
ax.annotate("", xy=(3.7, 3.8), xytext=(xs[0] + 0.2, ys[0][-1] + 1.35), arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, ls="--"))
ax.annotate("", xy=(8.3, 3.8), xytext=(xs[6] - 0.2, ys[6][-1] + 1.35), arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, ls="--"))
ax.set_xlim(-1.2, 13.2); ax.set_ylim(-3.4, 4.6); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "gen-autoencoder-neurons.svg"); plt.close(fig)
print("wrote", len(list(OUT.glob("gen-*.svg"))), "figures")
