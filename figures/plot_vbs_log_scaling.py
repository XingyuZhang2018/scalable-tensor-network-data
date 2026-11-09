from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
KAGOME_BOND_RANGE_CSV = DATA_DIR / "kagome_2x2_bond_abs_range_chi1024.csv"
OUTPUT_PDF = SCRIPT_DIR / "Kagome_2x2_bond_range_log_scaling.pdf"

GREEN = "#009E73"


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.65,
        "figure.dpi": 150,
        "savefig.dpi": 450,
    }
)


def load_kagome_bond_range_points() -> list[dict[str, float | int]]:
    with KAGOME_BOND_RANGE_CSV.open(newline="", encoding="utf-8") as handle:
        points = [
            {
                "D": int(row["D"]),
                "chi": int(row["chi"]),
                "xi": float(row["xi"]),
                "abs_bond_range": float(row["abs_bond_range"]),
            }
            for row in csv.DictReader(handle)
        ]
    if [int(point["D"]) for point in points] != [8, 9, 10]:
        raise ValueError("Expected Kagome 2x2 D=8,9,10 points")
    if any(int(point["chi"]) != 1024 for point in points):
        raise ValueError("Expected chi=1024 for every Kagome 2x2 point")
    return points


def fit_kagome_bond_range(
    points: list[dict[str, float | int]],
) -> tuple[float, float, float, float]:
    x = np.log(np.array([1.0 / float(point["xi"]) for point in points]))
    y = np.log(
        np.array([float(point["abs_bond_range"]) for point in points])
    )
    (slope, intercept), covariance = np.polyfit(x, y, 1, cov=True)
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total
    slope_error = float(np.sqrt(covariance[0, 0]))
    return float(slope), float(intercept), slope_error, r_squared


def plot() -> None:
    kagome_points = load_kagome_bond_range_points()
    (
        kagome_slope,
        kagome_intercept,
        kagome_slope_error,
        kagome_r_squared,
    ) = fit_kagome_bond_range(kagome_points)
    kagome_x = np.log(
        np.array([1.0 / float(point["xi"]) for point in kagome_points])
    )
    kagome_y = np.log(
        np.array(
            [float(point["abs_bond_range"]) for point in kagome_points]
        )
    )

    fig, ax_kagome = plt.subplots(figsize=(3.40, 2.30))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.96)
    ax_kagome.scatter(
        kagome_x,
        kagome_y,
        color=GREEN,
        marker="^",
        s=26,
        edgecolor="black",
        linewidth=0.40,
        zorder=3,
    )
    kagome_x_fit = np.linspace(float(kagome_x.min()), float(kagome_x.max()), 240)
    ax_kagome.plot(
        kagome_x_fit,
        kagome_slope * kagome_x_fit + kagome_intercept,
        color=GREEN,
        linewidth=0.95,
        zorder=2,
    )
    ax_kagome.text(
        0.04,
        0.95,
        rf"$\Delta_b={kagome_slope:.2f}\pm{kagome_slope_error:.2f}$",
        transform=ax_kagome.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    ax_kagome.set_xlabel(r"$\log(1/\xi)$")
    
    ax_kagome.set_ylabel(r"$\log(\Delta E_b)$")
    ax_kagome.tick_params(width=0.55, length=2.5, direction="out")
    ax_kagome.set_xticks([-0.70, -0.66, -0.62])
    ax_kagome.set_yticks([-3.94, -3.88, -3.82, -3.76])
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

    print(
        f"Kagome 2x2 points: {len(kagome_points)}; "
        f"D=8--10; chi=1024; slope={kagome_slope:.8f}+-"
        f"{kagome_slope_error:.8f}; R2={kagome_r_squared:.8f}"
    )
    print(f"saved {OUTPUT_PDF}")


if __name__ == "__main__":
    plot()
