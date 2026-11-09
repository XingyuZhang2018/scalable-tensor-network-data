"""Plot one representative projector-energy error per p for D=8 and D=16."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data" / "fig8_projector_total_energy_error_p_D8_D16.tsv"
PDF = ROOT / "figures" / "fig8_projector_total_energy_error_p_D8_D16.pdf"
PNG = ROOT / "tmp" / "pdfs" / "fig8_projector_total_energy_error_p_D8_D16.png"

MM = 1 / 25.4
SERIES = {
    8: {
        "color": "#0B5AA4",
        "marker": "o",
        "label": r"$D=8,\ \chi=128$ (direct)",
    },
    16: {
        "color": "#D55E00",
        "marker": "s",
        "label": r"$D=16,\ \chi=1024$ (extrapolated)",
    },
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "mathtext.fontset": "stixsans",
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with DATA.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            rows.append(
                {
                    "D": int(row["D"]),
                    "p": int(row["p"]),
                    "delta": float(row["delta_E_J1"]),
                }
            )
    return rows


def main() -> None:
    rows = load_rows()
    fig, ax = plt.subplots(figsize=(86 * MM, 62 * MM))

    for D in (8, 16):
        selected = sorted((row for row in rows if row["D"] == D), key=lambda row: row["p"])
        style = SERIES[D]
        ax.plot(
            [row["p"] for row in selected],
            [row["delta"] for row in selected],
            color=style["color"],
            linewidth=1.15,
            zorder=2,
        )
        ax.scatter(
            [row["p"] for row in selected],
            [row["delta"] for row in selected],
            marker=style["marker"],
            s=22,
            facecolor=style["color"],
            edgecolor=style["color"],
            linewidth=0.7,
            label=style["label"],
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(24, 5500)
    ax.set_ylim(3.8e-9, 1.45e-7)
    ax.axhline(1.0e-7, color="0.35", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlabel(r"oversampling $p$")
    ax.set_ylabel(r"total-energy error $|\Delta E|/J_1$")
    ticks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks], rotation=35, ha="right")
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", handletextpad=0.4)
    ax.grid(False)

    fig.tight_layout(pad=0.6)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
