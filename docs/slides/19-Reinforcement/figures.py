#!/usr/bin/env python3
"""Generated figures for the 19-Reinforcement deck (run with a Python that has
numpy and matplotlib: /usr/local/bin/python3.12 figures.py). Writes SVGs to
images/. Every figure comes from a small simulation in this file, seeded, so the
deck can be regenerated exactly.

  bandit-regret.svg          cumulative regret of epsilon-greedy over four servers
                             with unknown latency, epsilon in {0, 0.1, 0.3}
  bandit-best-arm.svg        fraction of rounds the truly fastest server is chosen;
                             pure greedy plateaus (it never tries the server it
                             underestimated)
  ucb-thompson.svg           UCB: estimate plus confidence bonus per server;
                             Thompson: one posterior per server
  abr-learning-curve.svg     tabular Q-learning on the ABR MDP, held-out QoE per
                             evaluation against the three baselines
  abr-qtable.svg             one slice of the learned Q-table (buffer bucket x
                             action at a middle throughput bucket)
  abr-discretization.svg     the same learner with 3, 6 and 12 buffer buckets
  abr-reward-penalty.svg     what the learned policy does when the rebuffer
                             penalty changes (average bitrate, rebuffering,
                             switches on the held-out trace)

The ABR environment follows the Hands-On 19 design: four bitrates, four-second
chunks, a synthetic throughput trace, state = (buffer bucket, throughput bucket,
last bitrate), reward = bitrate utility - 4.3 x rebuffer seconds - 1 x |bitrate
change| in Mbps (the linear QoE from the Pensieve paper), baselines fixed-lowest,
rate-based and buffer-based.
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
LIGHT = "#bbbbbb"


# ----------------------------------------------------------------------------
# Part 1: bandits over four servers
# ----------------------------------------------------------------------------
LATENCY_MS = np.array([120.0, 95.0, 80.0, 105.0])   # true mean latency per server
LATENCY_SD = 25.0                                    # noise on each request
SERVER_NAMES = ["server A", "server B", "server C", "server D"]


def run_bandit(epsilon, rounds, rng):
    """Epsilon-greedy over four servers. Reward is negative latency. Returns the
    per-round regret and the indicator 'chose the best server'."""
    best = LATENCY_MS.min()
    est = np.zeros(4)
    n = np.zeros(4)
    regret = np.empty(rounds)
    best_chosen = np.empty(rounds)
    for t in range(rounds):
        if rng.random() < epsilon:
            a = rng.integers(4)
        else:
            a = int(np.argmax(est)) if n.sum() > 0 else int(rng.integers(4))
        lat = LATENCY_MS[a] + LATENCY_SD * rng.standard_normal()
        n[a] += 1
        est[a] += (-lat - est[a]) / n[a]          # running mean of reward
        regret[t] = LATENCY_MS[a] - best           # expected regret this round
        best_chosen[t] = a == int(np.argmin(LATENCY_MS))
    return regret, best_chosen


def bandit_figures(rounds=1000, seeds=200):
    eps_list = [(0.0, ORANGE, "ε = 0 (pure greedy)"),
                (0.1, BLUE, "ε = 0.1"),
                (0.3, GREEN, "ε = 0.3")]
    curves = {}
    for eps, _, _ in eps_list:
        reg = np.zeros(rounds)
        best = np.zeros(rounds)
        for s in range(seeds):
            r, b = run_bandit(eps, rounds, np.random.default_rng(1000 + s))
            reg += np.cumsum(r)
            best += b
        curves[eps] = (reg / seeds, best / seeds)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for eps, color, label in eps_list:
        reg, _ = curves[eps]
        ax.plot(reg, color=color, lw=2.2, label=label)
        ax.annotate(f"{reg[-1]:,.0f} ms", (rounds - 1, reg[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11, color=GREY)
        print(f"eps={eps}: cumulative regret after {rounds} rounds = {reg[-1]:.0f} ms, "
              f"best-server rate in last 100 rounds = {curves[eps][1][-100:].mean():.2f}")
    ax.set_xlabel("requests (rounds)")
    ax.set_ylabel("cumulative regret (ms of latency above the best server)")
    ax.set_xlim(0, rounds * 1.12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.grid(True, color="#eeeeee")
    fig.tight_layout()
    fig.savefig(OUT / "bandit-regret.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    k = 25
    kern = np.ones(k) / k
    for eps, color, label in eps_list:
        _, best = curves[eps]
        sm = np.convolve(best, kern, mode="valid")
        ax.plot(np.arange(k - 1, rounds), sm, color=color, lw=2.2, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("requests (rounds)")
    ax.set_ylabel("fraction of rounds the fastest server is chosen")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(True, color="#eeeeee")
    fig.tight_layout()
    fig.savefig(OUT / "bandit-best-arm.svg")
    plt.close(fig)


def ucb_thompson_figure(t=60):
    rng = np.random.default_rng(7)
    n = np.array([30, 15, 10, 5])                 # pulls so far, per server
    est = -(LATENCY_MS + LATENCY_SD / np.sqrt(n) * rng.standard_normal(4))
    c = 2.0 * LATENCY_SD
    bonus = c * np.sqrt(np.log(t) / n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.0))
    x = np.arange(4)
    ax1.bar(x, est, color=[LIGHT] * 4, width=0.55, edgecolor=GREY)
    ax1.errorbar(x, est, yerr=[np.zeros(4), bonus], fmt="none", ecolor=PURPLE, elinewidth=2.4, capsize=8)
    for i in range(4):
        ax1.text(x[i], est[i] + bonus[i] + 2, f"n = {n[i]}", ha="center", fontsize=11, color=GREY)
    pick = int(np.argmax(est + bonus))
    ax1.scatter([pick], [est[pick] + bonus[pick] + 8], marker="v", s=120, color=PURPLE, zorder=3)
    ax1.text(pick + 0.42, est[pick] + bonus[pick] + 8, "UCB picks this", ha="left", va="center", fontsize=11, color=PURPLE)
    ax1.set_xticks(x, SERVER_NAMES)
    ax1.set_ylabel("estimated reward (−latency, ms)")
    ax1.set_ylim(-165, -25)
    ax1.set_title("UCB: estimate + bonus for uncertainty", fontsize=13, color=GREY)
    ax1.grid(True, axis="y", color="#eeeeee")

    grid = np.linspace(-170, -40, 400)
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    for i in range(4):
        sd = LATENCY_SD / np.sqrt(n[i])
        pdf = np.exp(-0.5 * ((grid - est[i]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax2.plot(grid, pdf, color=colors[i], lw=2.2, label=f"{SERVER_NAMES[i]} (n = {n[i]})")
        ax2.fill_between(grid, pdf, color=colors[i], alpha=0.12)
    ax2.set_xlabel("reward (−latency, ms)")
    ax2.set_ylabel("posterior density")
    ax2.set_yticks([])
    ax2.set_title("Thompson: sample one value per server, play the max", fontsize=13, color=GREY)
    ax2.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ucb-thompson.svg")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Parts 2 and 3: the ABR MDP and tabular Q-learning
# ----------------------------------------------------------------------------
BITRATES = np.array([0.3, 0.75, 1.2, 1.85])   # Mbps
CHUNK_S = 4.0
BUF_CAP_S = 40.0
CHUNKS_PER_EPISODE = 80
THR_EDGES = np.array([0.5, 1.0, 1.6])          # Mbps, 4 throughput buckets


def make_trace(rng, n=CHUNKS_PER_EPISODE):
    """Synthetic throughput in Mbps: a slow log-space random walk with a few
    abrupt level shifts, clipped to a plausible range."""
    level = np.log(rng.uniform(0.6, 2.0))
    out = np.empty(n)
    for i in range(n):
        if rng.random() < 0.04:
            level = np.log(rng.uniform(0.3, 3.0))
        level += 0.08 * rng.standard_normal()
        level = np.clip(level, np.log(0.2), np.log(3.5))
        out[i] = np.exp(level) * rng.uniform(0.85, 1.15)
    return out


class ABREnv:
    def __init__(self, buf_buckets=6, rebuf_penalty=4.3, switch_penalty=1.0):
        self.buf_edges = np.linspace(0, 24, buf_buckets)[1:]  # bucket edges in seconds
        self.n_buf = buf_buckets
        self.rebuf_penalty = rebuf_penalty
        self.switch_penalty = switch_penalty

    def n_states(self):
        return self.n_buf * (len(THR_EDGES) + 1) * len(BITRATES)

    def state(self, buf, thr, last):
        b = int(np.searchsorted(self.buf_edges, buf))
        t = int(np.searchsorted(THR_EDGES, thr))
        return (b * (len(THR_EDGES) + 1) + t) * len(BITRATES) + last

    def run(self, trace, policy):
        """policy(state_index, buffer_s, measured_thr, last_action) -> action.
        Returns (total QoE, average bitrate, rebuffer seconds, switches, trajectory)."""
        buf, last, thr_meas = 0.0, 0, trace[0]
        total, bits, rebuf, switches = 0.0, [], 0.0, 0
        traj = []
        for i, thr in enumerate(trace):
            s = self.state(buf, thr_meas, last)
            a = policy(s, buf, thr_meas, last)
            dl = BITRATES[a] * CHUNK_S / thr                  # seconds to fetch the chunk
            stall = max(0.0, dl - buf)
            buf = max(0.0, buf - dl) + CHUNK_S
            if buf > BUF_CAP_S:                                # wait for room
                buf = BUF_CAP_S
            r = BITRATES[a] - self.rebuf_penalty * stall - self.switch_penalty * abs(BITRATES[a] - BITRATES[last])
            thr_meas = BITRATES[a] * CHUNK_S / dl              # observed throughput
            s2 = self.state(buf, thr_meas, a)
            traj.append((s, a, r, s2))
            total += r
            bits.append(BITRATES[a])
            rebuf += stall
            switches += a != last
            last = a
        return total, float(np.mean(bits)), rebuf, switches, traj


def baseline_fixed(s, buf, thr, last):
    return 0


def baseline_rate(s, buf, thr, last):
    ok = np.where(BITRATES <= 0.9 * thr)[0]
    return int(ok[-1]) if len(ok) else 0


def baseline_buffer(s, buf, thr, last, reservoir=5.0, cushion=20.0):
    frac = np.clip((buf - reservoir) / (cushion - reservoir), 0, 1)
    return int(round(frac * (len(BITRATES) - 1)))


def q_learning(env, train_traces, held_out, episodes=5000, alpha=0.05, gamma=0.9,
               eval_every=50, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states(), len(BITRATES)))
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))        # decaying epsilon

        def pol(s, buf, thr, last):
            if rng.random() < eps:
                return int(rng.integers(len(BITRATES)))
            return int(np.argmax(Q[s]))

        trace = train_traces[rng.integers(len(train_traces))]
        _, _, _, _, traj = env.run(trace, pol)
        for s, a, r, s2 in traj:
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
        if ep % eval_every == 0 or ep == episodes - 1:
            greedy = lambda s, buf, thr, last: int(np.argmax(Q[s]))
            total = np.mean([env.run(tr, greedy)[0] for tr in held_out])
            curve.append((ep, total))
    return Q, np.array(curve)


def abr_figures():
    rng = np.random.default_rng(42)
    train_traces = [make_trace(rng) for _ in range(40)]
    held_out = [make_trace(np.random.default_rng(900 + i)) for i in range(5)]
    env = ABREnv()
    Q, curve = q_learning(env, train_traces, held_out)
    base = {name: np.mean([env.run(tr, p)[0] for tr in held_out])
            for name, p in [("fixed lowest", baseline_fixed), ("rate-based", baseline_rate),
                            ("buffer-based", baseline_buffer)]}
    print("held-out QoE:", {k: round(v, 1) for k, v in base.items()}, "Q-learning", round(curve[-1, 1], 1))

    # learning curve
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(curve[:, 0], curve[:, 1], color=PURPLE, lw=2.2, label="tabular Q-learning (greedy policy)")
    for (name, val), color in zip(base.items(), [GREY, BLUE, GREEN]):
        ax.axhline(val, color=color, lw=1.6, ls="--")
        ax.text(curve[-1, 0] * 1.01, val, name, va="center", fontsize=11, color=color)
    ax.set_xlabel("training episodes (one episode = one 80-chunk session)")
    ax.set_ylabel("QoE per session on held-out traces")
    ax.set_xlim(0, curve[-1, 0] * 1.3)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(True, color="#eeeeee")
    fig.tight_layout()
    fig.savefig(OUT / "abr-learning-curve.svg")
    plt.close(fig)

    # one slice of the Q-table: throughput bucket 2 (1.0 to 1.6 Mbps), last bitrate = 0.75
    t_bucket, last = 2, 1
    rows = np.array([Q[env.state(b, 1.3, last)] for b in np.r_[0.0, env.buf_edges + 0.01]])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    im = ax.imshow(rows, cmap="Purples", aspect="auto")
    for i in range(rows.shape[0]):
        j = int(np.argmax(rows[i]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.4))
        for jj in range(rows.shape[1]):
            ax.text(jj, i, f"{rows[i, jj]:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if rows[i, jj] > rows.max() * 0.6 else GREY)
    buf_labels = ["0–4.8", "4.8–9.6", "9.6–14.4", "14.4–19.2", "19.2–24", "24+"]
    ax.set_yticks(range(rows.shape[0]), [f"{l} s" for l in buf_labels])
    ax.set_xticks(range(4), [f"{b} Mbps" for b in BITRATES])
    ax.set_xlabel("action: next chunk's bitrate")
    ax.set_ylabel("state: buffer bucket")
    ax.set_title("Q(s, a) with throughput 1.0–1.6 Mbps, last bitrate 0.75 (box = greedy choice)",
                 fontsize=11, color=GREY)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "abr-qtable.svg")
    plt.close(fig)

    # discretization: 3, 6, 12 buffer buckets
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for nb, color in [(3, GREEN), (6, PURPLE), (12, ORANGE)]:
        e = ABREnv(buf_buckets=nb)
        _, c = q_learning(e, train_traces, held_out, seed=1)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=2.0, label=f"{nb} buffer buckets ({e.n_states()} states)")
        print(f"{nb} buckets: final held-out QoE {c[-1, 1]:.1f}")
    ax.axhline(base["buffer-based"], color=GREY, lw=1.4, ls="--")
    ax.text(5, base["buffer-based"] + 1, "buffer-based baseline", fontsize=10, color=GREY)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("QoE per session on held-out traces")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(True, color="#eeeeee")
    fig.tight_layout()
    fig.savefig(OUT / "abr-discretization.svg")
    plt.close(fig)

    # reward design: change the rebuffer penalty, retrain, compare behaviour
    settings = [(1.0, BLUE), (4.3, PURPLE), (20.0, ORANGE)]
    stats = []
    for pen, _ in settings:
        e = ABREnv(rebuf_penalty=pen)
        Qp, _ = q_learning(e, train_traces, held_out, seed=2)
        greedy = lambda s, buf, thr, last, Qp=Qp: int(np.argmax(Qp[s]))
        runs = [e.run(tr, greedy) for tr in held_out]
        stats.append((np.mean([r[1] for r in runs]), np.mean([r[2] for r in runs]), np.mean([r[3] for r in runs])))
        print(f"rebuffer penalty {pen}: avg bitrate {stats[-1][0]:.2f} Mbps, rebuffer {stats[-1][1]:.1f} s, switches {stats[-1][2]:.1f}")
    stats = np.array(stats)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    titles = ["average bitrate (Mbps)", "rebuffering per session (s)", "bitrate switches per session"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        x = np.arange(3)
        ax.bar(x, stats[:, k], color=[c for _, c in settings], width=0.6)
        for i in range(3):
            ax.text(x[i], stats[i, k] + stats[:, k].max() * 0.02, f"{stats[i, k]:.2f}" if k == 0 else f"{stats[i, k]:.1f}",
                    ha="center", fontsize=11, color=GREY)
        ax.set_xticks(x, [f"penalty {p:g}" for p, _ in settings])
        ax.set_title(title, fontsize=13, color=GREY)
        ax.set_ylim(0, stats[:, k].max() * 1.2)
        ax.grid(True, axis="y", color="#eeeeee")
    fig.suptitle("Same learner, same traces, three rebuffer penalties (held-out evaluation)",
                 fontsize=12, color=GREY, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "abr-reward-penalty.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bandit_figures()
    ucb_thompson_figure()
    abr_figures()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
