from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data" / "multigpu_scaling_flmap.csv"
OUT_STEM = SCRIPT_DIR / "multigpu_scaling"

SERIES_COLORS = {
    ("1D", "forward"): "#0072B2",
    ("1D", "backward"): "#D55E00",
    ("2D", "forward"): "#009E73",
    ("2D", "backward"): "#AA3377",
}
CASES = {(D, chi) for D in (8, 10, 12) for chi in (512, 1024)}
P_BY_METHOD = {"1D": (1, 2, 4, 8, 16), "2D": (1, 4, 8, 16)}
GPUS_PER_NODE = 4
MULTINODE_FILL = "#F3F3F3"
MULTINODE_TEXT = "#666666"


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def load_times() -> tuple[dict, dict]:
    rows = list(csv.DictReader(DATA_FILE.open(encoding="utf-8", newline="")))
    if len(rows) != 96:
        raise ValueError(f"expected 96 selected timings, got {len(rows)}")

    times: dict[tuple[int, int, int, str, str], float] = {}
    baselines: dict[tuple[int, int, int, str, str], float] = {}
    for row in rows:
        key = (
            int(row["P"]),
            int(row["D"]),
            int(row["chi"]),
            row["method"],
            row["phase"],
        )
        value = float(row["time_ms"])
        baseline = float(row["baseline_time_ms"])
        if (key in times or not np.isfinite(value) or value <= 0 or
                not np.isfinite(baseline) or baseline <= 0):
            raise ValueError(f"invalid or duplicate timing {key}: {value}")
        if key[0] == 1 and baseline != value:
            raise ValueError("one-GPU normalization anchor mismatch")
        times[key] = value
        baselines[key] = baseline

    expected = set()
    for P in (1, 2):
        for D, chi in CASES:
            for phase in ("forward", "backward"):
                expected.add((P, D, chi, "1D", phase))
    for P in (4, 8, 16):
        for D, chi in CASES:
            for method in ("1D", "2D"):
                for phase in ("forward", "backward"):
                    expected.add((P, D, chi, method, phase))
    if set(times) != expected:
        raise ValueError("timing-key set mismatch")
    return times, baselines


TIMES, BASELINES = load_times()


def speedup(method: str, phase: str, D: int, chi: int, P: int) -> float:
    if P == 1:
        return 1.0
    key = (P, D, chi, method, phase)
    # Each selected case retains its same-trial, same-width-policy reference.
    return BASELINES[key] / TIMES[key]


def case_series(method: str, phase: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    ps = np.asarray(P_BY_METHOD[method], dtype=float)
    rows = []
    for D, chi in sorted(CASES):
        values = np.asarray([speedup(method, phase, D, chi, int(P)) for P in ps])
        if metric == "efficiency":
            values = values / ps
        rows.append(values)
    return ps, np.vstack(rows)


def mean_series(method: str, phase: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    ps, values = case_series(method, phase, metric)
    return ps, np.mean(values, axis=0)


def band_style(color: str) -> dict:
    # Show translucent ranges without outlines or hatching.
    return dict(
        facecolor=to_rgba(color, 0.13),
        edgecolor="none",
        linewidth=0,
    )


def plot_mean(ax, metric: str, method: str, phase: str, color: str, marker: str) -> None:
    p, cases = case_series(method, phase, metric)
    values = np.mean(cases, axis=0)
    # This is the spread across the six selected (D, chi) cases, not timing error.
    ax.fill_between(
        p,
        np.min(cases, axis=0),
        np.max(cases, axis=0),
        **band_style(color),
        zorder=2,
    )
    ax.plot(
        p,
        values,
        color=color,
        linewidth=1.55,
        linestyle="-",
        marker=marker,
        markersize=4.0 if marker == "o" else 3.8,
        markerfacecolor=color if marker == "o" else "white",
        markeredgecolor=color,
        markeredgewidth=0.75,
        zorder=3,
        clip_on=False,
    )


def style_axis(ax, metric: str) -> None:
    ax.set_xlim(0.65, 16.5)
    # P > GPUS_PER_NODE spans several nodes; shade from midway between 4 and 8.
    x_node = 1.5 * GPUS_PER_NODE
    ax.axvspan(x_node, 16.5, color=MULTINODE_FILL, linewidth=0, zorder=0)
    y_label = 19.0 if metric == "speedup" else 1.285
    ax.text(x_node + 0.35, y_label, "multi-node", fontsize=5.9, color=MULTINODE_TEXT, va="center")
    ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    ax.grid(False, which="both", axis="both")
    ax.tick_params(width=0.55, length=2.5, direction="out")
    ax.set_xlabel("Number of GPUs, $P$")
    if metric == "speedup":
        ax.set_ylim(1, 20)
        ax.set_yticks([1, 4, 8, 12, 16, 20], labels=["1", "4", "8", "12", "16", "20"])
        ax.set_ylabel("Speedup")
        xref = np.linspace(1, 16, 200)
        ax.plot(xref, xref, color="0.30", linewidth=0.8, linestyle=(0, (1.5, 2.2)), zorder=1)
        ax.text(12.3, 13.0, "ideal: $y=x$", fontsize=5.7, color="0.30", rotation=31)
    else:
        ax.set_ylim(0, 1.34)
        ax.set_yticks(
            np.arange(0, 1.201, 0.2),
            labels=[f"{v:d}" for v in range(0, 121, 20)],
        )
        ax.set_ylabel("Parallel efficiency (%)")
        ax.axhline(1.0, color="0.30", linewidth=0.75, linestyle=(0, (1.5, 2.2)), zorder=1)
        ax.text(15.1, 1.02, "ideal", fontsize=5.7, color="0.30", ha="right")


def plot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.42))
    for ax, metric in zip(axes, ("speedup", "efficiency")):
        for method, marker in (("1D", "o"), ("2D", "D")):
            for phase in ("forward", "backward"):
                plot_mean(ax, metric, method, phase, SERIES_COLORS[method, phase], marker)
        style_axis(ax, metric)

    axes[0].text(-0.14, 1.045, "a", transform=axes[0].transAxes, fontsize=9.0, fontweight="bold", va="top")
    axes[1].text(-0.14, 1.045, "b", transform=axes[1].transAxes, fontsize=9.0, fontweight="bold", va="top")

    handles, labels = [], []
    for method, marker in (("1D", "o"), ("2D", "D")):
        for phase in ("forward", "backward"):
            color = SERIES_COLORS[method, phase]
            band = Patch(**band_style(color))
            line = Line2D([0], [0], color=color, marker=marker, markersize=3.8,
                          linewidth=1.35, linestyle="-",
                          markerfacecolor=color if method == "1D" else "white")
            handles.append((band, line))
            labels.append(f"{method}Slice {phase}")
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.9,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.97, wspace=0.25)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    for method in ("1D", "2D"):
        for phase in ("forward", "backward"):
            p, values = mean_series(method, phase, "speedup")
            print(
                f"{method} {phase}: "
                + ", ".join(f"P={int(P)} {value:.6f}x" for P, value in zip(p, values))
            )


if __name__ == "__main__":
    plot()
