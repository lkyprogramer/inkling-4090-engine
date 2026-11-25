# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""Speedup charts and result tables for Inkling 2 Mini on an H100 PCIe.

Mean single-stream decode tok/s over 32 prompts (two sets of 16), each engine through its
own OpenAI-compatible server: streaming /v1/completions on the chat-template token ids,
512 tokens, greedy, context 8,192. Usage: python make_charts.py [out_dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# Mean decode tok/s: (set A, set B) per engine and mode.
SETS = {
    "vllm_py": {"plain": (130.2, 130.3), "mtp": (218.0, 201.2)},
    "vllm_rs": {"plain": (130.4, 130.4), "mtp": (218.2, 201.2)},
    "sglang": {"plain": (135.6, 135.7), "mtp": (219.0, 192.8)},
    "astral": {"plain": (172.2, 171.8), "mtp": (242.0, 224.6)},
}
MEAN = {e: {m: sum(v) / 2 for m, v in modes.items()} for e, modes in SETS.items()}
NAMES = {"vllm_py": "vLLM (Python)", "vllm_rs": "vLLM (Rust frontend)", "sglang": "SGLang", "astral": "Astral Inference Server"}

BG, GRID, AXIS, INK, MUTED = "#EBEBEB", "#D2D2D2", "#7A7A7A", "#1F1F1F", "#555555"
COLOR = {"base": "#A6A6A6", "sglang": "#2F6FB3", "astral": "#76B900"}
FONT = "Inter"
plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 12})
META = {"Software": None, "Title": None, "Author": None, "Description": None, "Creation Time": None}


def chart(base: str, file: str) -> None:
    series = [(base, COLOR["base"]), ("sglang", COLOR["sglang"]), ("astral", COLOR["astral"])]
    groups = [("Without MTP", "plain"), ("With MTP (DSpark draft)", "mtp")]
    fig = plt.figure(figsize=(8.2, 6.4), dpi=200, facecolor=BG)
    ax = fig.add_axes((0.11, 0.2, 0.84, 0.56), facecolor=BG)
    width, gap = 0.24, 0.012
    for gi, (_, mode) in enumerate(groups):
        for si, (eng, col) in enumerate(series):
            v = MEAN[eng][mode] / MEAN[base][mode]
            x = gi + (si - 1) * (width + gap)
            ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, v, boxstyle="round,pad=0,rounding_size=0.012", linewidth=0, facecolor=col, mutation_aspect=1 / 6))
            ax.text(x, v + 0.035, f"{v:.2f}x", ha="center", va="bottom", fontsize=12.5, color=INK, fontweight="bold" if eng == "astral" else "normal")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([i * 0.25 for i in range(7)])
    ax.set_yticklabels(["0", "0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x"], color=MUTED, fontsize=11.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g for g, _ in groups], fontsize=13.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.2)
    # Legend: colored squares over the plot, as in a press chart.
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for _, c in series]
    fig.legend(handles, [NAMES[e] for e, _ in series], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=14, handlelength=1.1, handleheight=1.1, columnspacing=2.2, handletextpad=0.6)
    fig.text(0.5, 0.85, f"Inkling 2 Mini  ·  decode speed relative to {NAMES[base]}  ·  1 request, 32 prompts, H100 PCIe", ha="center", fontsize=10.5, color=MUTED)
    fig.text(0.5, 0.065, "Mean decode tok/s  ·  streaming /v1/completions  ·  512 tokens  ·  greedy  ·  context 8,192", ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.035, "MTP: vLLM and SGLang official DSpark (7 drafts)  ·  Astral adaptive DSpark (2–6 drafts)", ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(OUT / file, facecolor=BG, metadata=META)
    plt.close(fig)


def table(base: str, file: str) -> None:
    rows = [base, "sglang", "astral"]
    head = ["Engine", "Without MTP\n(tok/s)", "Speedup", "With MTP\n(tok/s)", "Speedup"]
    data = []
    for e in rows:
        p, m = MEAN[e]["plain"], MEAN[e]["mtp"]
        data.append([NAMES[e], f"{p:.1f}", f"{p / MEAN[base]['plain']:.2f}x", f"{m:.1f}", f"{m / MEAN[base]['mtp']:.2f}x"])
    fig = plt.figure(figsize=(8.2, 3.9), dpi=200, facecolor=BG)
    fig.text(0.05, 0.9, f"Inkling 2 Mini: results vs {NAMES[base]}", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.05, 0.825, "Mean single-stream decode speed over 32 prompts, H100 PCIe 80 GB", fontsize=10.5, color=MUTED)
    cols = [0.05, 0.43, 0.575, 0.72, 0.865]  # left edges
    widths = [0.38, 0.145, 0.145, 0.145, 0.085]
    top, hh, rh = 0.72, 0.14, 0.12
    ax = fig.add_axes((0, 0, 1, 1), facecolor="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.04, top - hh), 0.92, hh, facecolor="#3A3A3A", linewidth=0))
    for j, h in enumerate(head):
        x = cols[j] + (0.0 if j == 0 else widths[j] / 2 if j < 4 else 0.04)
        ax.text(x if j == 0 else (cols[j] + (0.0725 if j < 4 else 0.045)), top - hh / 2, h, ha="left" if j == 0 else "center", va="center", fontsize=10.5, color="white", fontweight="bold", linespacing=1.15)
    for i, (e, r) in enumerate(zip(rows, data)):
        y = top - hh - (i + 1) * rh
        fill = "#E3F1C8" if e == "astral" else ("#F6F6F6" if i % 2 == 0 else "#FFFFFF")
        ax.add_patch(Rectangle((0.04, y), 0.92, rh, facecolor=fill, linewidth=0))
        sw = COLOR["astral"] if e == "astral" else COLOR["sglang"] if e == "sglang" else COLOR["base"]
        ax.add_patch(Rectangle((0.05, y + rh / 2 - 0.02), 0.022, 0.04, facecolor=sw, linewidth=0))
        for j, cell in enumerate(r):
            bold = e == "astral"
            if j == 0:
                ax.text(0.083, y + rh / 2, cell, ha="left", va="center", fontsize=11.5, color=INK, fontweight="bold" if bold else "normal")
            else:
                ax.text(cols[j] + (0.0725 if j < 4 else 0.045), y + rh / 2, cell, ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold" if bold else "normal")
    fig.text(0.05, 0.075, "Without MTP: plain decoding.  With MTP: DSpark draft (vLLM / SGLang 7 drafts per round, Astral adaptive 2–6).", fontsize=9, color=MUTED)
    fig.text(0.05, 0.035, "Streaming /v1/completions, 512 tokens, greedy, context 8,192; speedup relative to the first row.", fontsize=9, color=MUTED)
    fig.savefig(OUT / file, facecolor=BG, metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chart("vllm_py", "astral_vs_vllm_python.png")
    table("vllm_py", "astral_vs_vllm_python_table.png")
    chart("vllm_rs", "astral_vs_vllm_rust.png")
    table("vllm_rs", "astral_vs_vllm_rust_table.png")
    print("written to", OUT)
