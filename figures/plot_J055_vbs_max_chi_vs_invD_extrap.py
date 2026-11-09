from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


SCRIPT_DIR = Path(__file__).resolve().parent
VBS_DATA_CSV = SCRIPT_DIR / "data" / "Square_J055_Fig6_max_chi_points.csv"
INV_XI_DATA_CSV = SCRIPT_DIR / "data" / "Square_J055_Fig6_invxi_points.csv"
OUTPUT_PDF = SCRIPT_DIR / "Square_J055_VBS_max_chi_vs_invD_extrap.pdf"
PREVIEW_DIR = SCRIPT_DIR.parent / "output" / "pdf"
PREVIEW_PNG = PREVIEW_DIR / "Square_J055_VBS_max_chi_vs_invD_extrap.png"

POINT_COLOR = "#0072B2"


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "font.family": "Arial",
        "font.size": 6.0,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 450,
    }
)


def load_points(path: Path, column: str) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            d = int(row["D"])
            points.append(
                {"D": d, "chi": int(row["chi"]), "inv_D": 1.0 / d, "value": float(row[column])}
            )
    points.sort(key=lambda point: int(point["D"]))
    return points


def fit_linear(points: list[dict[str, float | int | str]]) -> dict[str, float]:
    x = np.array([float(point["inv_D"]) for point in points])
    y = np.array([float(point["value"]) for point in points])
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([slope, intercept])
    rss = float(np.sum(residual**2))
    total = float(np.sum((y - y.mean()) ** 2))
    residual_variance = rss / (len(y) - 2)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "intercept_stderr": float(np.sqrt(covariance[1, 1])),
        "r_squared": float(1.0 - rss / total),
    }


def fit_power_law(points: list[dict[str, float | int | str]]) -> dict[str, float]:
    x = np.array([float(point["inv_D"]) for point in points])
    y = np.array([float(point["value"]) for point in points])

    def solve_at_alpha(alpha: float) -> tuple[float, float, float]:
        z = (8.0 * x) ** alpha
        design = np.column_stack((np.ones_like(z), z))
        intercept, amplitude_at_d8 = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ np.array([intercept, amplitude_at_d8])
        return float(intercept), float(amplitude_at_d8), float(residual @ residual)

    optimum = minimize_scalar(
        lambda alpha: solve_at_alpha(alpha)[2],
        bounds=(0.05, 8.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not optimum.success or not (0.051 < optimum.x < 7.999):
        raise ValueError("VBS power-law exponent did not have an interior optimum")
    alpha = float(optimum.x)
    intercept, amplitude_at_d8, rss = solve_at_alpha(alpha)
    if not (0.0 < intercept < min(y) and amplitude_at_d8 > 0.0):
        raise ValueError("VBS power-law fit violates positive-order-parameter assumptions")
    z = (8.0 * x) ** alpha
    jacobian = np.column_stack((np.ones_like(z), z, amplitude_at_d8 * z * np.log(8.0 * x)))
    covariance = (rss / (len(y) - 3)) * np.linalg.inv(jacobian.T @ jacobian)
    return {
        "intercept": intercept,
        "intercept_stderr": float(np.sqrt(covariance[0, 0])),
        "amplitude": float(amplitude_at_d8 * 8.0**alpha),
        "alpha": alpha,
        "alpha_stderr": float(np.sqrt(covariance[2, 2])),
        "r_squared": float(1.0 - rss / np.sum((y - y.mean()) ** 2)),
    }


def draw_fit(ax, fit: dict[str, float]) -> None:
    x_fit = np.linspace(0.0, 0.13, 260)
    y_fit = (
        fit["intercept"] + fit["amplitude"] * x_fit ** fit["alpha"]
        if "alpha" in fit
        else fit["slope"] * x_fit + fit["intercept"]
    )
    ax.plot(
        x_fit,
        y_fit,
        color="black",
        lw=0.75,
        ls=(0, (3.2, 2.0)),
        zorder=1,
    )


def draw_points(ax, points: list[dict[str, float | int | str]]) -> None:
    for point in points:
        ax.scatter(
            [float(point["inv_D"])],
            [float(point["value"])],
            s=18,
            marker="o",
            facecolor=POINT_COLOR,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )


def draw_intercept(ax, fit: dict[str, float]) -> None:
    ax.errorbar(
        [0.0],
        [fit["intercept"]],
        yerr=[fit["intercept_stderr"]],
        fmt="*",
        ms=6.5,
        mfc="black",
        mec="black",
        mew=0.35,
        ecolor="black",
        elinewidth=0.65,
        capsize=2.0,
        capthick=0.65,
        clip_on=False,
        zorder=4,
    )


def draw_panel(ax, points, *, ylabel: str, ylim: tuple[float, float], yticks):
    fit = fit_power_law(points) if ylabel == r"$O_{\rm VBS}$" else fit_linear(points)
    draw_fit(ax, fit)
    draw_points(ax, points)
    draw_intercept(ax, fit)
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 0.135)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.0, 0.05, 0.10])
    ax.set_yticks(yticks)
    return fit


def main() -> None:
    vbs_points = load_points(VBS_DATA_CSV, "P_vbs")
    inv_xi_points = load_points(INV_XI_DATA_CSV, "inv_xi_site")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(3.40, 1.80))
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.23, top=0.92, wspace=0.50)
    vbs_fit = draw_panel(
        axes[0],
        vbs_points,
        ylabel=r"$O_{\rm VBS}$",
        ylim=(0.0, 0.14),
        yticks=[0.0, 0.04, 0.08, 0.12],
    )
    inv_xi_fit = draw_panel(
        axes[1],
        inv_xi_points,
        ylabel=r"$1/\xi$",
        ylim=(0.0, 0.46),
        yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.28,
            1.01,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )
        axis.tick_params(width=0.5, length=2.2, direction="out")
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)

    figure.savefig(OUTPUT_PDF)
    figure.savefig(PREVIEW_PNG, dpi=450)
    plt.close(figure)

    print("single-column Figure 7; panels use their selected D=8,10,12,14,16 sources")
    print("D  chi_vbs  chi_xi      P_VBS             1/xi (sites^-1)")
    for vbs, inv_xi in zip(vbs_points, inv_xi_points):
        print(
            f"{int(vbs['D']):2d}  {int(vbs['chi']):7d}  {int(inv_xi['chi']):6d}  "
            f"{float(vbs['value']):.15f}  {float(inv_xi['value']):.15f}"
        )
    print(
        "VBS power-law fit: "
        f"intercept={vbs_fit['intercept']:.12f}; "
        f"stderr={vbs_fit['intercept_stderr']:.12f}; "
        f"alpha={vbs_fit['alpha']:.8f}; "
        f"alpha_stderr={vbs_fit['alpha_stderr']:.8f}; "
        f"R2={vbs_fit['r_squared']:.8f}"
    )
    print(
        "1/xi fit: "
        f"intercept={inv_xi_fit['intercept']:.12f}; "
        f"stderr={inv_xi_fit['intercept_stderr']:.12f}; "
        f"R2={inv_xi_fit['r_squared']:.8f}"
    )
    print(f"saved {OUTPUT_PDF}")
    print(f"saved {PREVIEW_PNG}")


if __name__ == "__main__":
    main()
