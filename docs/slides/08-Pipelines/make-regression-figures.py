#!/usr/bin/env python3
"""Figures for the regression-metrics slides in 08-Pipelines (residuals, MSE vs MAE, R²).

Synthetic data with fixed seeds; palette is the course dataviz default (blue, orange,
muted gray). Run from this directory: python3 make-regression-figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#333333"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRAY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "grid.color": "#e6e6e2", "axes.grid": True,
                     "axes.axisbelow": True})
rng = np.random.default_rng(7)


def residuals_figure():
    """Scatter, least-squares line, residual segments; MSE, RMSE, MAE on the figure."""
    x = rng.uniform(5, 60, 28)                     # packets per flow
    y = 1.4 * x + 6 + rng.normal(0, 7, 28)         # KB per flow
    w = np.polyfit(x, y, 1); yhat = np.polyval(w, x); e = y - yhat
    mse, mae = np.mean(e**2), np.mean(np.abs(e)); rmse = np.sqrt(mse)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.linspace(0, 65, 2); ax.plot(xs, np.polyval(w, xs), color=BLUE, lw=2, label="least-squares fit")
    for xi, yi, yh in zip(x, y, yhat):
        ax.plot([xi, xi], [yi, yh], color=ORANGE, lw=1.2, alpha=0.9, zorder=2)
    ax.scatter(x, y, s=40, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label="flows")
    ax.plot([], [], color=ORANGE, lw=1.2, label="residual  $e_i = y_i - \\hat{y}_i$")
    ax.set_xlabel("packets in flow"); ax.set_ylabel("bytes in flow (KB)")
    ax.set_xlim(0, 65); ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.04, f"MSE = {mse:.0f} KB²\nRMSE = {rmse:.1f} KB\nMAE = {mae:.1f} KB", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, lw=0.8))
    fig.tight_layout(); fig.savefig("images/regression-residuals.png", dpi=200); plt.close(fig)


def mse_vs_mae_figure():
    """Two clusters (ACK-like small flows, data-like large flows): the squared-error line
    chases the big cluster, the absolute-error line stays with the many small ones."""
    x_small = rng.uniform(2, 12, 60);  y_small = 0.08 * x_small + 0.3 + rng.normal(0, 0.15, 60)   # ACK-ish: tiny bytes
    x_big = rng.uniform(20, 40, 12);   y_big = 1.5 * x_big + rng.normal(0, 4, 12)                 # data: big bytes
    x = np.concatenate([x_small, x_big]); y = np.concatenate([y_small, y_big])
    w_ls = np.polyfit(x, y, 1)
    # least absolute deviation by iteratively reweighted least squares
    w_lad = w_ls.copy()
    for _ in range(60):
        r = np.abs(y - np.polyval(w_lad, x)); wts = 1 / np.maximum(r, 1e-3)
        A = np.vstack([x, np.ones_like(x)]).T * np.sqrt(wts)[:, None]
        w_lad = np.linalg.lstsq(A, y * np.sqrt(wts), rcond=None)[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.linspace(0, 42, 2)
    ax.scatter(x, y, s=34, color=GRAY, edgecolor="white", linewidth=0.8, zorder=3, label="flows (60 ACK-sized, 12 data)")
    ax.plot(xs, np.polyval(w_ls, xs), color=BLUE, lw=2, label="minimizes MSE (squared error)")
    ax.plot(xs, np.polyval(w_lad, xs), color=ORANGE, lw=2, label="minimizes MAE (absolute error)")
    ax.set_xlabel("packets in flow"); ax.set_ylabel("bytes in flow (KB)")
    ax.set_xlim(0, 42); ax.set_ylim(-6, 70)
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("squared error lets the 12 big\nflows pull the line", xy=(14, np.polyval(w_ls, 14)), xytext=(16, 36),
                color=BLUE, fontsize=11, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("absolute error stays with\nthe 60 small flows", xy=(9, np.polyval(w_lad, 9)), xytext=(13, -3),
                color=ORANGE, fontsize=11, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    fig.tight_layout(); fig.savefig("images/regression-mse-vs-mae.png", dpi=200); plt.close(fig)


def r2_figure():
    """Same RMSE, different R²: the baseline is predicting the mean."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=False)
    for ax, spread, title in zip(axes, (30, 6), ("y varies a lot", "y barely varies")):
        x = rng.uniform(5, 60, 30)
        y = spread / 55 * x + 10 + rng.normal(0, 5, 30)   # noise sd 5 in both panels
        w = np.polyfit(x, y, 1); yhat = np.polyval(w, x); e = y - yhat
        rmse = np.sqrt(np.mean(e**2)); r2 = 1 - np.sum(e**2) / np.sum((y - y.mean())**2)
        xs = np.linspace(0, 65, 2)
        ax.scatter(x, y, s=34, color=GRAY, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot(xs, np.polyval(w, xs), color=BLUE, lw=2, label="model")
        ax.axhline(y.mean(), color=ORANGE, lw=2, ls="--", label="predict the mean")
        ax.set_title(f"{title}:  RMSE = {rmse:.1f},  R² = {r2:.2f}", fontsize=12, color=INK, loc="left")
        ax.set_xlabel("packets in flow"); ax.set_xlim(0, 65)
        ax.set_ylim(y.mean() - 30, y.mean() + 30)
    axes[0].set_ylabel("bytes in flow (KB)"); axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("images/regression-r2.png", dpi=200); plt.close(fig)


residuals_figure(); mse_vs_mae_figure(); r2_figure()
print("wrote images/regression-residuals.png, regression-mse-vs-mae.png, regression-r2.png")
