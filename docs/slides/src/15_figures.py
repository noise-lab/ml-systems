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
