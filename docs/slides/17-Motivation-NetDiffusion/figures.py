"""Generated figures for the 17-Diffusion deck (matplotlib, SVG).

Run from the deck directory:  /usr/local/bin/python3.12 figures.py
Writes images/gen-*.svg. Every picture is drawn from code (no data files):

  gen-forward-noise.svg   a synthetic nPrint-style packet bitmap pushed through
                          the DDPM closed form x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps
                          at six timesteps, with the schedule curves underneath
  gen-reverse-chain.svg   the same states read in the generation direction, with
                          the "predict the noise, subtract it" step between them
  gen-nprint-bitmap.svg   a real IPv4+TCP header byte string for a short flow laid
                          out as nPrint bits: one row per packet, one column per bit
  gen-accuracy-gap.svg    the pcap-vs-NetFlow-vs-NetShare accuracy table as bars
  gen-headline-gains.svg  the three headline NetDiffusion-vs-NetShare numbers
"""
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.family": "Helvetica", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = Path(__file__).resolve().parent / "images"
BLUE, ORANGE, GREEN, RED, GREY, MAROON = "#1f5fa8", "#e0862a", "#1a8a4a", "#c8302e", "#888888", "#800000"
rng = np.random.default_rng(17)

# ---------------------------------------------------------------- nPrint bits
def ipv4_tcp_packet(src, dst, sport, dport, seq, ack, flags, payload_len, ttl=64, win=65535, ident=0):
    """Return (ipv4_header_bytes, tcp_header_bytes) with real field values."""
    total = 20 + 20 + payload_len
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, ident, 0x4000, ttl, 6, 0,
                     bytes(src), bytes(dst))
    csum = ip_checksum(ip)
    ip = ip[:10] + struct.pack("!H", csum) + ip[12:]
    tcp = struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 5 << 4, flags, win, 0, 0)
    return ip, tcp

def ip_checksum(b):
    s = sum(struct.unpack("!%dH" % (len(b) // 2), b))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF

def bits(b, nbits):
    out = np.full(nbits, -1, dtype=int)
    arr = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    out[:len(arr)] = arr
    return out

SYN, ACK, PSH, FIN = 0x02, 0x10, 0x08, 0x01
def tcp_flow(n_data=10):
    """SYN / SYN-ACK / ACK, then a few data+ack pairs, then FIN/ACK: nPrint rows."""
    c, s = (10, 0, 0, 23), (52, 94, 236, 48)        # client, a streaming CDN-looking address
    cs, ss = 1_000_000, 7_000_000
    pk = [((c, s, 51234, 443, cs, 0, SYN, 0)),
          ((s, c, 443, 51234, ss, cs + 1, SYN | ACK, 0)),
          ((c, s, 51234, 443, cs + 1, ss + 1, ACK, 0)),
          ((c, s, 51234, 443, cs + 1, ss + 1, PSH | ACK, 517))]
    cseq, sseq = cs + 518, ss + 1
    for i in range(n_data):
        pk.append((s, c, 443, 51234, sseq, cseq, PSH | ACK, 1400)); sseq += 1400
        if i % 2 == 1:
            pk.append((c, s, 51234, 443, cseq, sseq, ACK, 0))
    pk.append((s, c, 443, 51234, sseq, cseq, FIN | ACK, 0))
    pk.append((c, s, 51234, 443, cseq, sseq + 1, FIN | ACK, 0))
    rows = []
    for i, (src, dst, sp, dp, sq, ak, fl, pl) in enumerate(pk):
        ip, tcp = ipv4_tcp_packet(src, dst, sp, dp, sq, ak, fl, pl, ident=4000 + i,
                                  ttl=64 if src == c else 57, win=65535 if src == c else 29200)
        rows.append(np.concatenate([bits(ip, 480), bits(tcp, 480), np.full(64, -1), np.full(64, -1)]))
    return np.array(rows)

NPRINT_CMAP = ListedColormap(["#d9d9d9", "#d7301f", "#2ca25f"])   # -1 grey, 0 red, 1 green

def draw_nprint(ax, M, title=None):
    ax.imshow(M, cmap=NPRINT_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999999")
    if title: ax.set_title(title, fontsize=12)

# ---------------------------------------------------------------- 1. nPrint bitmap
M = tcp_flow()
fig, ax = plt.subplots(figsize=(12, 4.6))
draw_nprint(ax, M)
n = M.shape[0]
ax.set_ylabel("packets in the flow (one row each)", fontsize=12)
ax.set_yticks(range(n)); ax.set_yticklabels([f"pkt {i}" for i in range(n)], fontsize=9)
groups = [(0, 480, "IPv4 header (480 bits)"), (480, 960, "TCP header (480 bits)"),
          (960, 1088, "UDP · ICMP\n(64 each, absent)")]
for a, b, lab in groups:
    ax.axvline(a - 0.5, color="black", lw=1)
    ax.text((a + b) / 2, n - 0.2, lab, ha="center", va="top", fontsize=13, transform=ax.transData)
ax.axvline(1023.5, color="black", lw=0.6, ls=":")
ax.axvline(1087.5, color="black", lw=1)
ax.set_xlim(-0.5, 1087.5); ax.set_ylim(n + 1.6, -0.5)
# field callouts (bit offsets inside the IPv4 and TCP headers)
callouts = [(0, "version / IHL"), (64, "TTL"), (96, "src addr"), (128, "dst addr"),
            (480, "ports"), (512, "seq #"), (544, "ack #"), (584, "flags"), (640, "options (absent)")]
for x, lab in callouts:
    ax.annotate(lab, xy=(x + 8, -0.5), xytext=(x + 8, -1.6), fontsize=12, ha="left", rotation=30,
                arrowprops=dict(arrowstyle="-", color="#444444", lw=0.8), annotation_clip=False)
ax.set_ylim(n + 1.6, -3.6)
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2ca25f", label="1"), Patch(color="#d7301f", label="0"),
                   Patch(color="#d9d9d9", label="−1 (field absent)")],
          loc="upper right", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(1.0, 1.22))
fig.tight_layout()
fig.savefig(OUT / "gen-nprint-bitmap.svg"); plt.close(fig)
print("nprint bitmap:", M.shape, "rows x bits; absent bits:", int((M == -1).sum()))

# ---------------------------------------------------------------- 2. forward process
T = 1000
beta = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - beta)
x0 = tcp_flow(n_data=40).astype(float)             # 65 packets, more texture
x0 = x0[:64]
x0s = x0.copy(); x0s[x0s == -1] = 0.0               # map {-1,0,1} -> {0,0,1} for a scalar signal
x0s = 2 * x0s - 1                                   # centre in [-1, 1]
steps = [0, 25, 100, 300, 600, 999]
eps = rng.standard_normal(x0s.shape)
def xt(t):
    if t == 0: return x0s
    return np.sqrt(abar[t]) * x0s + np.sqrt(1 - abar[t]) * eps

fig = plt.figure(figsize=(12, 4.4))
gs = fig.add_gridspec(2, len(steps), height_ratios=[2.2, 1.3], hspace=0.45, wspace=0.12)
for i, t in enumerate(steps):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(xt(t), cmap="gray", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {t}" + ("  (x₀, real)" if t == 0 else ("  (x_T, pure noise)" if t == 999 else "")),
                 fontsize=11)
    ax.set_xlabel(f"√ᾱ = {np.sqrt(abar[t]):.2f}   √(1−ᾱ) = {np.sqrt(1-abar[t]):.2f}" if t else "", fontsize=9)
ax = fig.add_subplot(gs[1, :])
ax.plot(np.sqrt(abar), color=BLUE, lw=2, label="√ᾱₜ  (how much of x₀ survives)")
ax.plot(np.sqrt(1 - abar), color=ORANGE, lw=2, label="√(1−ᾱₜ)  (how much noise is mixed in)")
ax.plot(beta / beta.max(), color=GREY, lw=1.2, ls="--", label="βₜ (linear schedule, rescaled)")
for t in steps[1:]:
    ax.axvline(t, color="#bbbbbb", lw=0.8, ls=":")
ax.set_xlim(0, T); ax.set_ylim(0, 1.02); ax.set_xlabel("timestep t  (T = 1000, β from 1e-4 to 0.02 as in DDPM)")
ax.legend(fontsize=9, loc="center right", frameon=False, ncol=1)
fig.savefig(OUT / "gen-forward-noise.svg", bbox_inches="tight"); plt.close(fig)
print("forward: abar at", {t: round(float(abar[t]), 3) for t in steps})

# ---------------------------------------------------------------- 3. reverse chain
rsteps = [999, 600, 300, 100, 25, 0]
fig, axes = plt.subplots(1, len(rsteps), figsize=(12, 2.9))
for ax, t in zip(axes, rsteps):
    ax.imshow(xt(t), cmap="gray", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("x_T ~ N(0, I)" if t == 999 else ("x₀ (generated)" if t == 0 else f"x_{t}"), fontsize=12)
fig.subplots_adjust(wspace=0.5, top=0.78, bottom=0.2)
for i in range(len(rsteps) - 1):
    a, b = axes[i], axes[i + 1]
    pa, pb = a.get_position(), b.get_position()
    y = (pa.y0 + pa.y1) / 2
    arr = FancyArrowPatch((pa.x1 + 0.004, y), (pb.x0 - 0.004, y), transform=fig.transFigure,
                          arrowstyle="-|>", mutation_scale=16, color=MAROON, lw=1.8)
    fig.patches.append(arr)
    fig.text((pa.x1 + pb.x0) / 2, y + 0.06, "εθ(xₜ, t)", ha="center", fontsize=10, color=MAROON)
    fig.text((pa.x1 + pb.x0) / 2, y - 0.09, "predict noise,\nsubtract it", ha="center", fontsize=8, color="#444444")
fig.text(0.5, 0.02, "one network, called once per step, each call a small supervised denoising task; "
         "a little fresh noise is re-injected after every step except the last",
         ha="center", fontsize=10, color="#444444")
fig.savefig(OUT / "gen-reverse-chain.svg", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4. accuracy gap (paper Table 1 / legacy s07-i05)
labels = ["pcap\nreal → real", "NetFlow\nreal → real", "NetShare NetFlow\nsynthetic → real",
          "NetShare NetFlow\nreal → synthetic"]
macro = [1.00, 0.86, 0.396, 0.503]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
cols = [GREEN, BLUE, RED, RED]
bars = ax.bar(labels, macro, color=cols, width=0.62)
for b, v in zip(bars, macro):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)
ax.set_ylim(0, 1.12); ax.set_ylabel("macro-level accuracy (best model)")
ax.set_title("Service classification: train → test data (Jiang et al. 2024, Table 1)", fontsize=12)
ax.tick_params(axis="x", labelsize=10)
fig.tight_layout(); fig.savefig(OUT / "gen-accuracy-gap.svg"); plt.close(fig)
print("accuracy gap:", dict(zip([l.replace(chr(10), ' ') for l in labels], macro)))

# ---------------------------------------------------------------- 5. headline gains
labels = ["statistical similarity,\ncommon header fields\n(JSD / EMD)",
          "statistical similarity,\nall pcap header fields",
          "ML accuracy,\ntrain on synthetic,\ntest on real"]
gains = [30, 70, 65]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.barh(labels, gains, color=[BLUE, BLUE, GREEN], height=0.55)
for b, v in zip(bars, gains):
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"> {v}%", va="center", fontsize=14, fontweight="bold")
ax.set_xlim(0, 90); ax.set_xlabel("improvement of NetDiffusion over NetShare (%)")
ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=11)
ax.set_title("Headline results, NetDiffusion vs. NetShare (SIGMETRICS 2024)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "gen-headline-gains.svg"); plt.close(fig)
print("headline gains:", dict(zip(["common", "all", "ml"], gains)))
