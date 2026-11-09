"""Plot stacked Heisenberg extrapolations for a single-column paper figure.

Panel (a): E/site versus 1/xi^3 (top) and 1/L^3 (bottom).
Panel (b): m^2 versus 1/xi (top) and 1/L (bottom).

Only one iPEPS trend is fitted in each panel: an ordinary least-squares fit to
every optimized iPEPS observation in the table (D=12 and D=14).  The QMC curves use
the finite-size forms reported in Sandvik, J. Stat. Mech. 2026, 043101.  Higher-order energy coefficients
and the tiny magnetization L^-5 coefficient are reproduced from the rounded published
table.  All input numbers live in data/heisenberg_qmc_ipeps_extrapolation_*.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUT_STEM = SCRIPT_DIR / "heisenberg_qmc_ipeps_extrapolation_single_column"
MM_TO_IN = 1.0 / 25.4
FIGSIZE = (89 * MM_TO_IN, 132 * MM_TO_IN)

# Match figure (c) in J2_results_stacked: compact Arial text, blue iPEPS,
# black finite-L comparison, no background grid, and thin publication-style axes.
FONT_SIZE = 7.0
AXES_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
LEGEND_SIZE = 5.8
FIT_LINEWIDTH = 1.05
REFERENCE_LINEWIDTH = 1.05
POINT_SIZE = 18

IPEPS_POINT_COLORS = {12: "C0", 14: "C2"}
IPEPS_MARKERS = {12: "o", 14: "D"}
IPEPS_POINT_SIZES = {}
IPEPS_LABELS = {}      # every row is labelled "$D=<value>$"
IPEPS_FIT_COLOR = "navy"
QMC_COLOR = "black"
QMC_EXTRAP_COLOR = QMC_COLOR

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def load_ipeps_points(path: Path) -> np.ndarray:
    """Load the optimized iPEPS observations (D, chi, xi, E/site, m)."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                (
                    int(row["D"]),
                    int(row["chi"]),
                    float(row["xi"]),
                    float(row["E"]),
                    float(row["m"]),
                )
            )
    return np.array(
        rows,
        dtype=[("D", "i4"), ("chi", "i4"), ("xi", "f8"), ("energy", "f8"), ("m", "f8")],
    )


def load_qmc_table(path: Path) -> np.ndarray:
    """Load the finite-size QMC table; blank energies become NaN."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            energy = row["E"].strip()
            rows.append(
                (
                    float(row["L"]),
                    float(energy) if energy else np.nan,
                    float(row["m2"]),
                )
            )
    return np.array(rows, dtype=[("L", "f8"), ("energy", "f8"), ("m2", "f8")])


def load_qmc_parameters(path: Path) -> dict[str, float]:
    """Load the published thermodynamic-limit values and fit coefficients."""
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["name"]: float(row["value"]) for row in csv.DictReader(stream)}


# D, chi, xi, E/site, m
IPEPS = load_ipeps_points(DATA_DIR / "heisenberg_qmc_ipeps_extrapolation_ipeps.csv")

# QMC 2026, Table 1: L, E/site, m_s^2.  The L=128,192,256 m_s^2 values
# are the additional data quoted in Eq. (19) and used in the magnetization fit.
QMC_TABLE = load_qmc_table(DATA_DIR / "heisenberg_qmc_ipeps_extrapolation_qmc_table.csv")

# Reported QMC thermodynamic-limit and low-order coefficients.
QMC_PARAMETERS = load_qmc_parameters(
    DATA_DIR / "heisenberg_qmc_ipeps_extrapolation_qmc_parameters.csv"
)
QMC_E_INF = QMC_PARAMETERS["E_inf"]
QMC_E_INF_ERROR = QMC_PARAMETERS["E_inf_error"]
QMC_E3 = QMC_PARAMETERS["e3"]
QMC_E4 = QMC_PARAMETERS["e4"]
QMC_M_INF = QMC_PARAMETERS["m_inf"]
QMC_M_INF_ERROR = QMC_PARAMETERS["m_inf_error"]
QMC_M1 = QMC_PARAMETERS["m1"]
QMC_MU = QMC_PARAMETERS["mu"]
QMC_GAMMA = QMC_PARAMETERS["gamma"]
QMC_XI0 = QMC_PARAMETERS["xi0"]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, R^2, and the one-SE intercept uncertainty."""
    matrix = np.column_stack((x, np.ones_like(x)))
    (slope, intercept), *_ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    dof = len(x) - 2
    x_centered = x - np.mean(x)
    sxx = float(np.sum(x_centered**2))
    residual_variance = ss_res / dof
    intercept_error = np.sqrt(
        residual_variance * (1.0 / len(x) + float(np.mean(x)) ** 2 / sxx)
    )
    return float(slope), float(intercept), float(r2), float(intercept_error)


def reproduce_qmc_energy_higher_orders() -> tuple[float, float]:
    """Reproduce e5 and e6 using the rounded L>=12 published table values."""
    rows = QMC_TABLE[np.isfinite(QMC_TABLE["energy"]) & (QMC_TABLE["L"] >= 12)]
    length = rows["L"]
    leading = QMC_E_INF + QMC_E3 / length**3 + QMC_E4 / length**4
    design = np.column_stack((length**-5, length**-6))
    (e5, e6), *_ = np.linalg.lstsq(design, rows["energy"] - leading, rcond=None)
    return float(e5), float(e6)


def reproduce_qmc_m5() -> float:
    """Reproduce the small L^-5 term with other parameters fixed as reported."""
    length = QMC_TABLE["L"]
    leading = (
        QMC_M_INF**2
        + QMC_M1 / length
        + QMC_MU * np.log(length / QMC_XI0) ** QMC_GAMMA / length**2
    )
    basis = length**-5
    return float(np.dot(basis, QMC_TABLE["m2"] - leading) / np.dot(basis, basis))


QMC_E5, QMC_E6 = reproduce_qmc_energy_higher_orders()
QMC_M5 = reproduce_qmc_m5()


def qmc_energy_from_inv_l3(inv_l3: np.ndarray) -> np.ndarray:
    """QMC energy form through L^-6, evaluated at x=1/L^3."""
    x = np.asarray(inv_l3, dtype=float)
    return (
        QMC_E_INF
        + QMC_E3 * x
        + QMC_E4 * x ** (4.0 / 3.0)
        + QMC_E5 * x ** (5.0 / 3.0)
        + QMC_E6 * x**2
    )


def qmc_m2_from_inv_l(inv_l: np.ndarray) -> np.ndarray:
    """QMC log-corrected m_s^2 form, including the reproduced L^-5 term."""
    x = np.asarray(inv_l, dtype=float)
    result = np.full_like(x, QMC_M_INF**2)
    positive = x > 0.0
    xp = x[positive]
    result[positive] = (
        QMC_M_INF**2
        + QMC_M1 * xp
        + QMC_MU * np.log(1.0 / (QMC_XI0 * xp)) ** QMC_GAMMA * xp**2
        + QMC_M5 * xp**5
    )
    return result


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False, which="both", axis="both")
    ax.tick_params(axis="both", width=0.8, length=3.5, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def put_ipeps_axis_on_top(ax: plt.Axes, label: str) -> None:
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x",
        colors=IPEPS_FIT_COLOR,
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=2,
    )
    ax.set_xlabel(label, color=IPEPS_FIT_COLOR, labelpad=3)
    ax.spines["top"].set_color(IPEPS_FIT_COLOR)


def put_qmc_axis_on_bottom(ax: plt.Axes, label: str) -> None:
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.tick_params(
        axis="x",
        colors=QMC_COLOR,
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        pad=2,
    )
    ax.set_xlabel(label, color=QMC_COLOR, labelpad=3)


def combined_legend(ax: plt.Axes, twin: plt.Axes, location: str = "lower left") -> None:
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = twin.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc=location,
        ncol=2,
        columnspacing=1.1,
        handletextpad=0.7,
        labelspacing=0.7,
        borderpad=0.65,
        framealpha=0.86,
    )


def parenthesized_uncertainty(value: float, error: float, decimals: int) -> str:
    """Format value and one-sigma error as value(last-digit uncertainty)."""
    error_digits = int(round(error * 10**decimals))
    return f"{value:.{decimals}f}({error_digits})"


def draw_extrapolation_guide(
    ax: plt.Axes,
    value: float,
    error: float,
    color: str,
    label: str | None,
    x_fraction: float,
    label_offset: float = 3.0,
    vertical_alignment: str = "bottom",
) -> None:
    """Draw a dashed extrapolated value and a translucent one-sigma band."""
    ax.axhspan(
        value - error,
        value + error,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.axhline(
        value,
        color=color,
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
    )
    if label is not None:
        ax.annotate(
            label,
            xy=(x_fraction, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            color=color,
            fontsize=5.5,
            zorder=8,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )


def draw_aligned_energy_label(
    ax: plt.Axes,
    method: str,
    value_text: str,
    value: float,
    color: str,
    label_offset: float,
    vertical_alignment: str,
) -> None:
    """Align the energy formula/value column across the iPEPS and QMC rows."""
    formula_x = 0.43
    common = {
        "xy": (formula_x, value),
        "xycoords": ("axes fraction", "data"),
        "textcoords": "offset points",
        "va": vertical_alignment,
        "color": color,
        "fontsize": 5.5,
        "zorder": 8,
        "bbox": {
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
    }
    ax.annotate(method, xytext=(-2, label_offset), ha="right", **common)
    ax.annotate(rf"$E_\infty={value_text}$", xytext=(2, label_offset), ha="left", **common)


def draw_aligned_magnetization_label(
    ax: plt.Axes,
    method: str,
    value_text: str,
    value: float,
    color: str,
    label_offset: float,
    vertical_alignment: str,
) -> None:
    """Stack the QMC and iPEPS magnetization results in one aligned column."""
    formula_x = 0.50
    common = {
        "xy": (formula_x, value),
        "xycoords": ("axes fraction", "data"),
        "textcoords": "offset points",
        "va": vertical_alignment,
        "color": color,
        "fontsize": 5.5,
        "zorder": 8,
        "bbox": {
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
    }
    ax.annotate(method, xytext=(-2, label_offset), ha="right", **common)
    ax.annotate(rf"$m={value_text}$", xytext=(2, label_offset), ha="left", **common)


def plot_energy(ax: plt.Axes) -> tuple[float, float, float, float]:
    inv_xi3 = 1.0 / IPEPS["xi"] ** 3
    energy = IPEPS["energy"]
    slope, intercept, r2, intercept_error = linear_fit(inv_xi3, energy)

    for bond_dimension in sorted({int(d) for d in IPEPS["D"]}):
        mask = IPEPS["D"] == bond_dimension
        ax.scatter(
            inv_xi3[mask],
            energy[mask],
            color=IPEPS_POINT_COLORS[bond_dimension],
            marker=IPEPS_MARKERS[bond_dimension],
            s=IPEPS_POINT_SIZES.get(bond_dimension, POINT_SIZE),
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
            label=IPEPS_LABELS.get(bond_dimension, rf"$D={bond_dimension}$"),
        )

    x_max = float(inv_xi3.max()) * 1.08
    x_fit = np.linspace(0.0, x_max, 200)
    ax.plot(
        x_fit,
        intercept + slope * x_fit,
        color=IPEPS_FIT_COLOR,
        linestyle="--",
        linewidth=FIT_LINEWIDTH,
        alpha=0.88,
        zorder=4,
        label="iPEPS fit",
    )
    qmc_ax = ax.twiny()
    qmc_tail = QMC_TABLE[np.isfinite(QMC_TABLE["energy"]) & (QMC_TABLE["L"] >= 64)]
    inv_l3 = 1.0 / qmc_tail["L"] ** 3
    qmc_ax.scatter(
        inv_l3,
        qmc_tail["energy"],
        color=QMC_COLOR,
        marker="s",
        s=16,
        edgecolor="white",
        linewidth=0.4,
        zorder=6,
        label="QMC data",
    )

    qmc_x_max = 4.0e-6
    qmc_x_fit = np.linspace(0.0, qmc_x_max, 300)
    qmc_ax.plot(
        qmc_x_fit,
        qmc_energy_from_inv_l3(qmc_x_fit),
        color=QMC_COLOR,
        linestyle=":",
        linewidth=REFERENCE_LINEWIDTH,
        alpha=0.88,
        zorder=4,
        label="QMC fit",
    )
    fractional_left = 0.04
    ax.set_xlim(-fractional_left * x_max, x_max)
    qmc_ax.set_xlim(-fractional_left * qmc_x_max, qmc_x_max)
    y_values = np.concatenate((energy, qmc_tail["energy"], [intercept, QMC_E_INF]))
    y_span = float(np.ptp(y_values))
    ax.set_ylim(float(y_values.min() - 0.06 * y_span), float(y_values.max() + 0.06 * y_span))
    qmc_ax.set_ylim(ax.get_ylim())

    draw_extrapolation_guide(
        ax,
        intercept,
        intercept_error,
        IPEPS_FIT_COLOR,
        None,
        0.46,
    )
    draw_extrapolation_guide(
        ax,
        QMC_E_INF,
        QMC_E_INF_ERROR,
        QMC_EXTRAP_COLOR,
        None,
        0.78,
    )
    draw_aligned_energy_label(
        ax,
        "iPEPS:",
        parenthesized_uncertainty(intercept, intercept_error, 9),
        intercept,
        IPEPS_FIT_COLOR,
        label_offset=1.5,
        vertical_alignment="bottom",
    )
    draw_aligned_energy_label(
        ax,
        "QMC:",
        parenthesized_uncertainty(QMC_E_INF, QMC_E_INF_ERROR, 9),
        QMC_E_INF,
        QMC_EXTRAP_COLOR,
        label_offset=-2.5,
        vertical_alignment="top",
    )

    ax.set_xticks(np.arange(0.0, 5.1e-4, 1.0e-4))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
    qmc_ax.set_xticks(np.arange(0.0, 4.1e-6, 1.0e-6))
    qmc_ax.set_xticklabels(["0", "1", "2", "3", "4"])
    put_ipeps_axis_on_top(ax, r"$1/\xi^{3}$  ($\times 10^{-4}$)")
    put_qmc_axis_on_bottom(qmc_ax, r"$1/L^{3}$  ($\times 10^{-6}$)")

    ax.set_ylabel(r"$E$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    qmc_ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    qmc_ax.spines["top"].set_visible(False)
    qmc_ax.spines["bottom"].set_linewidth(0.8)
    style_axes(ax)
    combined_legend(ax, qmc_ax, location="lower left")
    return slope, intercept, r2, intercept_error


def plot_magnetization(ax: plt.Axes) -> tuple[float, float, float, float]:
    inv_xi = 1.0 / IPEPS["xi"]
    m2 = IPEPS["m"] ** 2
    slope, intercept, r2, intercept_error = linear_fit(inv_xi, m2)

    for bond_dimension in sorted({int(d) for d in IPEPS["D"]}):
        mask = IPEPS["D"] == bond_dimension
        ax.scatter(
            inv_xi[mask],
            m2[mask],
            color=IPEPS_POINT_COLORS[bond_dimension],
            marker=IPEPS_MARKERS[bond_dimension],
            s=IPEPS_POINT_SIZES.get(bond_dimension, POINT_SIZE),
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
            label=IPEPS_LABELS.get(bond_dimension, rf"$D={bond_dimension}$"),
        )

    x_max = float(inv_xi.max()) * 1.08
    x_fit = np.linspace(0.0, x_max, 200)
    ax.plot(
        x_fit,
        intercept + slope * x_fit,
        color=IPEPS_FIT_COLOR,
        linestyle="--",
        linewidth=FIT_LINEWIDTH,
        alpha=0.88,
        zorder=4,
        label="iPEPS fit",
    )
    qmc_ax = ax.twiny()
    qmc_tail = QMC_TABLE[QMC_TABLE["L"] >= 96]
    inv_l = 1.0 / qmc_tail["L"]
    qmc_ax.scatter(
        inv_l,
        qmc_tail["m2"],
        color=QMC_COLOR,
        marker="s",
        s=16,
        edgecolor="white",
        linewidth=0.4,
        zorder=6,
        label="QMC data",
    )

    qmc_x_max = 0.011
    qmc_x_fit = np.linspace(0.0, qmc_x_max, 300)
    qmc_ax.plot(
        qmc_x_fit,
        qmc_m2_from_inv_l(qmc_x_fit),
        color=QMC_COLOR,
        linestyle=":",
        linewidth=REFERENCE_LINEWIDTH,
        alpha=0.88,
        zorder=4,
        label="QMC fit",
    )
    fractional_left = 0.04
    ax.set_xlim(-fractional_left * x_max, x_max)
    qmc_ax.set_xlim(-fractional_left * qmc_x_max, qmc_x_max)
    y_values = np.concatenate((m2, qmc_tail["m2"], [intercept, QMC_M_INF**2]))
    y_span = float(np.ptp(y_values))
    ax.set_ylim(float(y_values.min() - 0.14 * y_span), float(y_values.max() + 0.06 * y_span))
    qmc_ax.set_ylim(ax.get_ylim())

    ipeps_m = float(np.sqrt(intercept))
    ipeps_m_error = intercept_error / (2.0 * ipeps_m)
    qmc_m2_error = 2.0 * QMC_M_INF * QMC_M_INF_ERROR
    draw_extrapolation_guide(
        ax,
        intercept,
        intercept_error,
        IPEPS_FIT_COLOR,
        None,
        0.50,
    )
    draw_extrapolation_guide(
        ax,
        QMC_M_INF**2,
        qmc_m2_error,
        QMC_EXTRAP_COLOR,
        None,
        0.50,
    )
    draw_aligned_magnetization_label(
        ax,
        "QMC:",
        parenthesized_uncertainty(QMC_M_INF, QMC_M_INF_ERROR, 6),
        QMC_M_INF**2,
        QMC_EXTRAP_COLOR,
        label_offset=0.5,
        vertical_alignment="bottom",
    )
    draw_aligned_magnetization_label(
        ax,
        "iPEPS:",
        parenthesized_uncertainty(ipeps_m, ipeps_m_error, 5),
        intercept,
        IPEPS_FIT_COLOR,
        label_offset=-2.5,
        vertical_alignment="top",
    )

    ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08])
    qmc_ax.set_xticks([0.0, 0.0025, 0.0050, 0.0075, 0.0100])
    qmc_ax.set_xticklabels(["0", "0.0025", "0.0050", "0.0075", "0.0100"])
    put_ipeps_axis_on_top(ax, r"$1/\xi$")
    put_qmc_axis_on_bottom(qmc_ax, r"$1/L$")

    ax.set_ylabel(r"$m^2$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    qmc_ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    qmc_ax.spines["top"].set_visible(False)
    qmc_ax.spines["bottom"].set_linewidth(0.8)
    style_axes(ax)
    # The encodings are identical in both panels; panel (a) carries the shared legend.
    return slope, intercept, r2, intercept_error


def build(output_stem: Path) -> None:
    fig, (energy_ax, magnetization_ax) = plt.subplots(2, 1, figsize=FIGSIZE)

    energy_slope, energy_intercept, energy_r2, energy_intercept_error = plot_energy(energy_ax)
    m2_slope, m2_intercept, m2_r2, m2_intercept_error = plot_magnetization(magnetization_ax)

    fig.subplots_adjust(left=0.210, right=0.985, bottom=0.10, top=0.90, hspace=0.55)
    for label, panel_ax in (("a", energy_ax), ("b", magnetization_ax)):
        panel_ax.text(
            -0.10,
            1.17,
            label,
            transform=panel_ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="top",
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(
        f"iPEPS energy fit: E_inf={energy_intercept:.12f}, "
        f"SE={energy_intercept_error:.3e}, slope={energy_slope:.12f}, R^2={energy_r2:.8f}"
    )
    print(
        f"iPEPS m^2 fit: m2_inf={m2_intercept:.12f}, "
        f"SE={m2_intercept_error:.3e}, m_inf={np.sqrt(m2_intercept):.12f}, "
        f"m_inf_SE={m2_intercept_error / (2.0 * np.sqrt(m2_intercept)):.3e}, slope={m2_slope:.12f}, "
        f"R^2={m2_r2:.8f}"
    )
    print(f"Reproduced QMC higher orders: e5={QMC_E5:.8f}, e6={QMC_E6:.8f}, m5={QMC_M5:.8f}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUT_STEM,
        help="Output path without extension (default: next to this script).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.output_stem.resolve())
