"""Rebuild Figure S6's error and speedup panels from the nested p/q scan."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PQ_DATA = ROOT / "figures/fig8_pq_scan_D8_chi128.tsv"
PDF = ROOT / "figures/fig8_pq_scan_D8_chi128.pdf"
PNG = ROOT / "tmp/pdfs/figS6_projector_validation.png"
COLORS = ("#005BAB", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "mathtext.fontset": "stixsans",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    pq_rows = read_rows(PQ_DATA)
    # Keep the physical panel dimensions and spacing of the three-panel layout.
    fig = plt.figure(figsize=(120.6 / 25.4, 61 / 25.4))
    axes = [fig.add_axes([left / 120.6, 0.18, 43.92 / 120.6, 0.775])
            for left in (14.76, 74.16)]

    # (a): the original error scan, with the original baseline and residual.
    ax = axes[0]
    baseline = next(r for r in pq_rows if int(r["p"]) == 0 and int(r["q"]) == 0)
    for q in (0, 1, 2):
        selected = sorted((r for r in pq_rows if int(r["q"]) == q), key=lambda r: int(r["p"]))
        ps = [int(r["p"]) for r in selected]
        ax.plot(ps, [float(r["max_abs_error"]) for r in selected],
                color=COLORS[q], marker=MARKERS[q], markersize=3.8, linewidth=1.1,
                label=rf"$q={q}$, worst")
        ax.plot(ps, [float(r["median_abs_error"]) for r in selected],
                color=COLORS[q], linestyle="--", linewidth=0.85,
                label=rf"$q={q}$, median")
    ax.axhline(float(baseline["max_abs_error"]), color="0.3", linestyle=":",
               linewidth=0.8, label=r"$p=0,\ q=0$")
    ax.axhline(float(baseline["boundary_err"]), color="0.65", linestyle=":",
               linewidth=0.8, label="VUMPS residual")
    ax.set(xlim=(-12, 268), ylim=(5e-11, 2.5e-7), yscale="log",
           xlabel=r"oversampling $p$", ylabel=r"$J_2$ bond $|\Delta|/J_1$")
    ax.set_xticks([0, 32, 64, 128, 256])
    ax.legend(frameon=False, fontsize=4.8, loc="lower left", bbox_to_anchor=(0, 0.10), ncol=2,
              columnspacing=0.8, handlelength=2.2, handletextpad=0.5,
              borderaxespad=0.7, labelspacing=0.4)

    # (b): retain the speedup scan; the redundant contraction-time panel is omitted.
    ax = axes[1]
    for q in (0, 1, 2):
        selected = sorted((r for r in pq_rows if int(r["q"]) == q), key=lambda r: int(r["p"]))
        ax.plot([int(r["p"]) for r in selected], [float(r["speedup"]) for r in selected],
                color=COLORS[q], marker=MARKERS[q], markersize=3.8, linewidth=1.1)
    ax.set(xlim=(-12, 268), ylim=(2.8, 18.6),
           xlabel=r"oversampling $p$", ylabel="speedup over direct")
    ax.set_xticks([0, 32, 64, 128, 256])
    ax.set_yticks(range(4, 19, 2))

    for ax, letter in zip(axes, "ab"):
        ax.tick_params(which="both", top=True, right=True)
        ax.text(0.97, 0.96, letter, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.0, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1})
        ax.grid(False)

    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=300)
    plt.close(fig)
    print(f"Saved {PDF}")
    print(f"Saved {PNG}")


if __name__ == "__main__":
    main()
