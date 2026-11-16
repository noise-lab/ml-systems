"""Generated figures for the 16-Clustering deck (matplotlib, SVG).

Run from docs/slides:  python3 src/16_figures.py
Writes 16-Clustering/images/gen-*.svg. Replaces the board sketches recovered
from the legacy deck with drawn-from-data versions of the same pictures.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from scipy.stats import norm, gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons

plt.rcParams.update({"svg.fonttype": "none", "font.family": "Helvetica", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = Path(__file__).resolve().parent.parent / "16-Clustering" / "images"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f5fa8", "#e0862a", "#1a8a4a", "#c8302e", "#888888", "#7b4ea3"
COLS = [BLUE, ORANGE, GREEN, PURPLE]
rng = np.random.default_rng(7)

def clean(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    if title: ax.set_title(title, fontsize=13)

# --- 1. k-means step by step: six frames
X, _ = make_blobs(n_samples=240, centers=[[-3, 2], [2.5, 2.5], [0, -2.5]], cluster_std=0.9, random_state=4)
C = np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]])   # deliberately bad seeds, all in the middle
frames = []
for it in range(3):
    lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1); frames.append(("assign", lab, C.copy()))
    Cn = np.array([X[lab == k].mean(0) for k in range(3)]); frames.append(("update", lab, (C.copy(), Cn.copy()))); C = Cn
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=100)
titles = ["1. random seeds", "2. E-step: assign to nearest seed", "3. M-step: move centers to the means",
          "4. E-step again: some points switch", "5. M-step again", "6. nothing moves: converged"]
for i, (ax, t) in enumerate(zip(axes.ravel(), titles)):
    kind, lab, c = frames[i]
    if i == 0: ax.scatter(X[:, 0], X[:, 1], s=12, color=GREY); ax.scatter(*c.T, marker="X", s=160, c=COLS[:3], edgecolor="k", zorder=5)
    elif kind == "assign": ax.scatter(X[:, 0], X[:, 1], s=12, c=[COLS[k] for k in lab]); ax.scatter(*c.T, marker="X", s=160, c=COLS[:3], edgecolor="k", zorder=5)
    else:
        old, new = c; ax.scatter(X[:, 0], X[:, 1], s=12, c=[COLS[k] for k in lab])
        for k in range(3): ax.annotate("", xy=new[k], xytext=old[k], arrowprops=dict(arrowstyle="->", color="k", lw=2))
        ax.scatter(*new.T, marker="X", s=160, c=COLS[:3], edgecolor="k", zorder=5)
    clean(ax, t)
fig.tight_layout(); fig.savefig(OUT / "gen-kmeans-steps.svg"); plt.close(fig)

# --- 2. elbow plot
sse = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in range(1, 9)]
fig, ax = plt.subplots(figsize=(7, 4.4), dpi=100)
ax.plot(range(1, 9), sse, "o-", color=BLUE, lw=2.5, ms=8)
ax.annotate("elbow at K = 3", xy=(3, sse[2]), xytext=(4.2, sse[0] * 0.55), fontsize=13, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax.set_xlabel("K (number of clusters)"); ax.set_ylabel("SSE (within-cluster squared distance)"); ax.set_title("The elbow method on the three-blob data")
fig.tight_layout(); fig.savefig(OUT / "gen-elbow.svg"); plt.close(fig)
print("SSE by K:", [round(s) for s in sse])

# --- 3. k-means on elongated clusters (failure)
Xe, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=1); Xe = Xe @ np.array([[0.6, -0.6], [-0.3, 0.8]])
labe = KMeans(3, n_init=10, random_state=0).fit_predict(Xe)
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
ax.scatter(Xe[:, 0], Xe[:, 1], s=14, c=[COLS[k] for k in labe]); clean(ax, "K-means (K = 3) on elongated clusters: the straight cuts miss the structure")
fig.tight_layout(); fig.savefig(OUT / "gen-kmeans-elongated.svg"); plt.close(fig)

# --- 4. two shapes k-means cannot split: rings; line plus two outliers
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
t = rng.uniform(0, 2 * np.pi, 220); r_in = rng.normal(0.7, 0.12, 220); r_out = rng.normal(2.4, 0.15, 220)
Xr = np.vstack([np.c_[r_in * np.cos(t), r_in * np.sin(t)], np.c_[r_out * np.cos(t), r_out * np.sin(t)]])
labr = KMeans(2, n_init=10, random_state=0).fit_predict(Xr)
axes[0].scatter(Xr[:, 0], Xr[:, 1], s=12, c=[COLS[k] for k in labr]); clean(axes[0], "Concentric rings: K = 2 cuts both rings in half")
xl = np.linspace(-4, 4, 150); Xl = np.c_[xl, 0.5 * xl + rng.normal(0, 0.25, 150)]; Xl = np.vstack([Xl, [[-2.5, 2.8], [3.0, -2.6]]])
labl = KMeans(2, n_init=10, random_state=0).fit_predict(Xl)
axes[1].scatter(Xl[:, 0], Xl[:, 1], s=12, c=[COLS[k] for k in labl]); axes[1].scatter(Xl[-2:, 0], Xl[-2:, 1], s=90, facecolor="none", edgecolor=RED, lw=2)
clean(axes[1], "A line plus two outliers: K = 2 bisects the line, keeps the outliers")
fig.tight_layout(); fig.savefig(OUT / "gen-kmeans-two-shapes.svg"); plt.close(fig)

# --- 5. GMM: observed density vs. the three components
x = np.linspace(0, 1500, 800); comps = [(0.35, 60, 18), (0.25, 120, 30), (0.40, 800, 180)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=100)
mix = sum(w * norm.pdf(x, m, s) for w, m, s in comps)
axes[0].plot(x, mix, color=BLUE, lw=3); axes[0].set_title("What you observe: one bumpy density")
axes[0].set_xlabel("packet length (bytes)"); axes[0].set_yticks([])
for (w, m, s), col, name in zip(comps, [PURPLE, GREEN, BLUE], ["scan, $\\pi_1$ = 0.35", "DNS, $\\pi_2$ = 0.25", "web, $\\pi_3$ = 0.40"]):
    axes[1].plot(x, w * norm.pdf(x, m, s), color=col, lw=2.5, label=f"{name}: μ = {m}, σ = {s}")
axes[1].plot(x, mix, color=GREY, lw=1.2, ls="--", label="mixture = $\\sum_k \\pi_k\\, p(x \\mid k)$")
axes[1].set_title("The story behind it: three Gaussians, priors $\\pi_k$"); axes[1].set_xlabel("packet length (bytes)"); axes[1].set_yticks([]); axes[1].legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "gen-gmm-mixture.svg"); plt.close(fig)

# --- 6. EM: known vs. unknown sources
pts_a = rng.normal(2, 0.8, 40); pts_b = rng.normal(5.5, 1.0, 40); xx = np.linspace(-1, 9, 400)
fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=100, sharex=True)
ax = axes[0]
ax.plot(xx, norm.pdf(xx, pts_a.mean(), pts_a.std()), color=RED, lw=2.5); ax.plot(xx, norm.pdf(xx, pts_b.mean(), pts_b.std()), color=BLUE, lw=2.5)
ax.scatter(pts_a, np.full_like(pts_a, -0.03), color=RED, s=30, clip_on=False); ax.scatter(pts_b, np.full_like(pts_b, -0.03), color=BLUE, s=30, clip_on=False)
ax.set_title("Sources known: fit each Gaussian from its own points (a mean and a variance each)"); ax.set_yticks([]); ax.set_ylim(-0.05, 0.6)
ax = axes[1]
allp = np.r_[pts_a, pts_b]; ga, gb = (1.0, 1.2), (6.5, 1.2)   # a rough current guess
pa = norm.pdf(allp, *ga); pb = norm.pdf(allp, *gb); resp = pa / (pa + pb)
ax.plot(xx, norm.pdf(xx, *ga), color=RED, lw=2.5, ls="--"); ax.plot(xx, norm.pdf(xx, *gb), color=BLUE, lw=2.5, ls="--")
for p_, r in zip(allp, resp):
    ax.plot([p_, p_], [-0.03, -0.03 + 0.12 * r], color=RED, lw=2); ax.plot([p_, p_], [-0.03 + 0.12 * r, 0.09], color=BLUE, lw=2)
ax.scatter(allp, np.full_like(allp, -0.03), color=GREY, s=30, clip_on=False)
ax.set_title("Sources unknown (gray points), current guess dashed: the E-step colors each point fractionally, then the M-step refits")
ax.set_yticks([]); ax.set_ylim(-0.05, 0.6); ax.set_xlabel("x")
fig.tight_layout(); fig.savefig(OUT / "gen-em-known-unknown.svg"); plt.close(fig)

# --- 7. distribution-based vs density-based shapes
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
Xb, lb = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=3)
axes[0].scatter(Xb[:, 0], Xb[:, 1], s=12, c=[COLS[k] for k in lb]); clean(axes[0], "Distribution-based: round blobs, a Gaussian per cluster")
Xm, lm = make_moons(n_samples=300, noise=0.06, random_state=3); core = rng.normal([0.5, 0.25], 0.08, (60, 2))
axes[1].scatter(Xm[:, 0], Xm[:, 1], s=12, c=[COLS[k] for k in lm]); axes[1].scatter(core[:, 0], core[:, 1], s=12, color=GREEN)
clean(axes[1], "Density-based: crescents around a core; any connected dense region is a cluster")
fig.tight_layout(); fig.savefig(OUT / "gen-distribution-vs-density.svg"); plt.close(fig)

# --- 8. DBSCAN: KDE density with peaks and outliers (1-D intuition)
samp = np.r_[rng.normal(2, 0.4, 120), rng.normal(5, 0.5, 150), rng.normal(8.5, 0.35, 90), [0.2, 11.0, 11.6]]
kde = gaussian_kde(samp); xs = np.linspace(-1, 13, 600)
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
dens = kde(xs); ax.plot(xs, dens, color="k", lw=2)
for lo, hi, col in [(0.8, 3.3, BLUE), (3.6, 6.5, ORANGE), (7.4, 9.6, GREEN)]:
    m = (xs > lo) & (xs < hi); ax.fill_between(xs[m], 0, dens[m], color=col, alpha=0.5)
ax.scatter([0.2, 11.0, 11.6], [0.005] * 3, marker="*", s=180, color=RED, zorder=5, label="outliers: no dense neighborhood")
ax.scatter(samp[:-3], np.full(len(samp) - 3, -0.012), s=6, color=GREY, clip_on=False)
ax.set_yticks([]); ax.set_xticks([]); ax.set_xlabel("feature value"); ax.set_title("Density view: peaks are clusters, the sparse tails are noise"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "gen-dbscan-density.svg"); plt.close(fig)

# --- 9. DBSCAN drawn from data: two clusters, eps balls, noise
Xd, _ = make_blobs(n_samples=[110, 60], centers=[[-2, 0], [2.5, 1.5]], cluster_std=[0.6, 0.45], random_state=5)
Xd = np.vstack([Xd, [[4.5, -2.2], [5.0, -1.6], [4.2, -1.4]]]); eps = 0.75
labd = DBSCAN(eps=eps, min_samples=5).fit_predict(Xd)
fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=100)
for k, col in [(0, BLUE), (1, ORANGE)]: ax.scatter(Xd[labd == k, 0], Xd[labd == k, 1], s=16, color=col)
ax.scatter(Xd[labd == -1, 0], Xd[labd == -1, 1], s=60, marker="x", color=RED, lw=2, label="noise (never reachable)")
for idx in [np.argmin(Xd[:, 0]), np.argmax(Xd[:, 1]), len(Xd) - 2]:
    ax.add_patch(Circle(Xd[idx], eps, fill=False, color="k", lw=1.5, ls="--"))
ax.annotate("ε-ball around a core point:\n≥ minPts neighbors, so the\ncluster grows through it", xy=Xd[np.argmin(Xd[:, 0])] + [eps, 0], xytext=(-4.4, 2.6), fontsize=11, arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("ε-ball around a stray point:\nempty, so it stays noise", xy=Xd[-2] + [-eps, 0], xytext=(0.3, -2.9), fontsize=11, arrowprops=dict(arrowstyle="->", lw=1))
clean(ax, f"DBSCAN (ε = {eps}, minPts = 5): two clusters found, three points left as noise"); ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "gen-dbscan.svg"); plt.close(fig)
print("DBSCAN labels:", dict(zip(*np.unique(labd, return_counts=True))))

# --- 10. dendrogram with a cut
pts = np.array([[0, 0], [0.3, 0.2], [0.1, 0.5], [3, 3], [3.3, 3.1], [3.1, 3.5], [6, 0], [6.2, 0.3], [8.5, 8.5]])
Zl = linkage(pts, "average")
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
cut = 3.0
dendrogram(Zl, ax=ax, labels=[f"p{i+1}" for i in range(9)], color_threshold=cut, above_threshold_color=GREY, leaf_font_size=12)
ax.axhline(cut, color=RED, ls="--", lw=2); ax.text(0.5, cut + 0.25, "cut here: 4 clusters (3, 3, 2, 1)", color=RED, fontsize=12)
ax.set_ylabel("merge distance (average linkage)"); ax.set_title("A dendrogram on nine points; the cut chooses K afterwards")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "gen-dendrogram-cut.svg"); plt.close(fig)
print("wrote", len(list(OUT.glob("gen-*.svg"))), "figures")

# --- 11. the board version of hierarchical clustering: nested loops on the points, beside the dendrogram
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import fcluster
pts2 = np.array([[1.0, 6.0], [2.2, 6.3], [0.6, 4.6], [4.6, 3.2], [5.8, 3.0], [3.4, 4.2], [9.0, 6.2], [10.0, 5.4], [2.0, 11.5]])
Z2 = linkage(pts2, "average")
fig, (axp, axd) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=100, gridspec_kw={"width_ratios": [1, 1.2]})
axp.scatter(pts2[:, 0], pts2[:, 1], s=70, color="k", zorder=5)
for i, (x, y) in enumerate(pts2):
    axp.annotate(str(i + 1), (x, y), xytext=(7, 5), textcoords="offset points", fontsize=15, fontweight="bold")
members = {i: [i] for i in range(len(pts2))}
for step, (a, b, dist, _) in enumerate(Z2):
    group = members[int(a)] + members[int(b)]; members[len(pts2) + step] = group
    if dist > 4.7: continue                      # the board stops at the three-cluster loop
    P = pts2[group]; cx, cy = P.mean(0); w, h = np.ptp(P, 0) + 1.5 + 0.35 * step
    axp.add_patch(Ellipse((cx, cy), w, h, fill=False, lw=2 if dist < 3 else 2.4, color=RED if dist < 3 else ORANGE))
axp.set_xlim(-1.5, 12.0); axp.set_ylim(0.5, 13.0); clean(axp, "Nine points; every loop is one merge")
dendrogram(Z2, ax=axd, labels=[str(i + 1) for i in range(9)], color_threshold=3.0, above_threshold_color=GREY, leaf_font_size=14)
axd.axhline(3.0, color=RED, ls="--", lw=2); axd.text(0.3, 3.2, "cut: 4 clusters  {1,2,3} {4,5,6} {7,8} {9}", color=RED, fontsize=12.5)
axd.axhline(4.7, color=ORANGE, ls="--", lw=2); axd.text(0.3, 4.9, "cut: 3 clusters  {1..6} {7,8} {9}", color=ORANGE, fontsize=12.5)
axd.set_ylabel("merge distance (average linkage)"); axd.set_title("The same merges as a dendrogram; read down from a cut")
axd.spines["top"].set_visible(False); axd.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "gen-hierarchical-board.svg"); plt.close(fig)
print("merge order:", [(int(a) + 1 if a < 9 else f"c{int(a)}", int(b) + 1 if b < 9 else f"c{int(b)}", round(float(d), 2)) for a, b, d, _ in Z2])
print("4 clusters:", fcluster(Z2, 4, "maxclust"), " 3 clusters:", fcluster(Z2, 3, "maxclust"))
