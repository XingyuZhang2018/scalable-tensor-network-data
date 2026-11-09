from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_kagome_merge_compact import draw_on_axes as draw_kagome_ansatz


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
STEM = "Square_J1J2_J05_vs_HB_Kagome_log_m2_vs_log_inv_xi_kagome_chi_ge_3D2"
DATA_CSV = DATA_DIR / f"{STEM}.csv"
BENCHMARK_CSV = DATA_DIR / "kagome_energy_benchmarks_arxiv_2512_18831.csv"
PESS_MAGNETIZATION_CSV = DATA_DIR / "kagome_3pess_su_min_magnetization.csv"
TWO_BY_TWO_ENERGY_CSV = DATA_DIR / "kagome_2x2_optimized_energy_xi.csv"
BOND_CSV = DATA_DIR / "kagome_unit_cell_bonds.csv"
MAGNETIZATION_CSV = DATA_DIR / "kagome_unit_cell_magnetization.csv"
ENERGY_OUT_STEM = SCRIPT_DIR / "Kagome_energy_xi3_unit_cells"
SCALING_OUT_STEM = SCRIPT_DIR / "Square_Kagome_log_m2_vs_log_inv_xi"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
BLUISH_GREEN = "#009E73"


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "font.family": "Arial",
        "font.size": 6.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.6,
        "axes.linewidth": 0.65,
        "figure.dpi": 150,
        "savefig.dpi": 450,
    }
)


def load_points() -> dict[str, list[dict[str, float | int | str]]]:
    points: dict[str, list[dict[str, float | int | str]]] = {
        "Square": [],
        "Kagome": [],
    }
    with DATA_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            system = row["system"]
            xi_log = float(row["xi"])
            # Both data sets advance by a physical distance of two nearest-
            # neighbour bonds per transfer step: the square data use a 2x2
            # cell, while Kagome(:merge) uses primitive vectors of length 2.
            transfer_period = 2.0
            xi = transfer_period * xi_log
            points[system].append(
                {
                    "system": system,
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "E": float(row["E"]),
                    "xi": xi,
                    "log_inv_xi": -np.log(xi),
                    "log_M2": float(row["log_M2"]),
                }
            )
    return points


def load_energy_benchmarks() -> list[dict[str, float | int | str]]:
    benchmarks: list[dict[str, float | int | str]] = []
    with BENCHMARK_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            benchmarks.append(
                {
                    "method": row["method"],
                    "D": int(row["D"]),
                    "E": float(row["E"]),
                }
            )
    return benchmarks


def load_optimized_two_by_two_energies() -> list[
    dict[str, float | int | str]
]:
    points: list[dict[str, float | int | str]] = []
    with TWO_BY_TWO_ENERGY_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append(
                {
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "E": float(row["E"]),
                    "xi": float(row["xi"]),
                }
            )
    return points


def select_two_by_two_max_chi_points(
    points: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    selected: list[dict[str, float | int | str]] = []
    for bond_dimension in (8, 9, 10):
        matching = [
            point for point in points if int(point["D"]) == bond_dimension
        ]
        if not matching:
            raise ValueError(f"Missing 2x2 data for D={bond_dimension}")
        selected.append(max(matching, key=lambda point: int(point["chi"])))
    return selected


def select_lowest_energy_points(
    points: list[dict[str, float | int | str]], count: int
) -> list[dict[str, float | int | str]]:
    if len(points) < count:
        raise ValueError(
            f"Requested {count} fit points, but only {len(points)} are available"
        )
    return sorted(points, key=lambda point: float(point["E"]))[:count]


def select_lowest_magnetization_points(
    points: list[dict[str, float | int | str]], count: int
) -> list[dict[str, float | int | str]]:
    if len(points) < count:
        raise ValueError(
            f"Requested {count} fit points, but only {len(points)} are available"
        )
    return sorted(points, key=lambda point: float(point["log_M2"]))[:count]


def load_bonds() -> list[dict[str, float | int | str]]:
    bonds: list[dict[str, float | int | str]] = []
    with BOND_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bonds.append(
                {
                    "unit_cell": row["unit_cell"],
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "i": int(row["i"]),
                    "j": int(row["j"]),
                    "bond": row["bond"],
                    "E": float(row["E"]),
                }
            )
    return bonds


def load_magnetizations() -> list[dict[str, float | int | str]]:
    magnetizations: list[dict[str, float | int | str]] = []
    with MAGNETIZATION_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            magnetizations.append(
                {
                    "unit_cell": row["unit_cell"],
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "i": int(row["i"]),
                    "j": int(row["j"]),
                    "site": int(row["site"]),
                    "Mnorm": float(row["Mnorm"]),
                    "Mx": float(row["Mx"]),
                    "Mz": float(row["Mz"]),
                }
            )
    return magnetizations


def fit_line(
    points: list[dict[str, float | int | str]],
) -> tuple[float, float, float]:
    x = np.array([float(point["log_inv_xi"]) for point in points])
    y = np.array([float(point["log_M2"]) for point in points])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return float(slope), float(intercept), r2


def fit_kagome_energy(
    points: list[dict[str, float | int | str]],
) -> tuple[float, float, float, float]:
    x = np.array([1.0 / float(point["xi"]) ** 3 for point in points])
    y = np.array([float(point["E"]) for point in points])
    (slope, intercept), covariance = np.polyfit(x, y, 1, cov=True)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    intercept_error = float(np.sqrt(covariance[1, 1]))
    return float(slope), float(intercept), intercept_error, r2


def extrapolate_kagome_energy(
    points: list[dict[str, float | int | str]],
) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("The energy extrapolation requires at least two points")
    x = np.array([1.0 / float(point["xi"]) ** 3 for point in points])
    y = np.array([float(point["E"]) for point in points])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def estimate_two_by_two_finite_chi_sensitivity(
    points: list[dict[str, float | int | str]],
    selected: list[dict[str, float | int | str]],
    central_intercept: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    companion_points: list[dict[str, float | int | str]] = []
    energy_error = np.zeros((2, len(selected)))

    for index, selected_point in enumerate(selected):
        lower_chi = [
            point
            for point in points
            if int(point["D"]) == int(selected_point["D"])
            and int(point["chi"]) < int(selected_point["chi"])
        ]
        if not lower_chi:
            raise ValueError(
                "A next-lower-chi companion is required for each 2x2 fit point"
            )
        companion = max(lower_chi, key=lambda point: int(point["chi"]))
        companion_points.append(companion)

        selected_energy = float(selected_point["E"])
        companion_energy = float(companion["E"])

        energy_error[0, index] = max(
            0.0, selected_energy - companion_energy
        )
        energy_error[1, index] = max(
            0.0, companion_energy - selected_energy
        )

    alternative_intercepts = [
        extrapolate_kagome_energy(list(choice))[1]
        for choice in product(
            *[
                (selected_point, companion_point)
                for selected_point, companion_point in zip(
                    selected, companion_points, strict=True
                )
            ]
        )
    ]
    intercept_error = (
        central_intercept - min(alternative_intercepts),
        max(alternative_intercepts) - central_intercept,
    )
    return energy_error, intercept_error


SQRT3 = np.sqrt(3.0)
BOND_ENDPOINTS = {
    "bond_12_energy": ((0, 0, 1), (0, 0, 2)),
    "bond_23_energy": ((0, 0, 2), (0, 0, 3)),
    "bond_31H_energy": ((0, 0, 3), (0, 1, 1)),
    "bond_32H_energy": ((0, 0, 3), (0, 1, 2)),
    "bond_31V_energy": ((0, 0, 3), (1, 0, 1)),
    "bond_21V_energy": ((0, 0, 2), (1, 0, 1)),
}

# A central kagome hexagon and the six outward triangle apices.  Modulo a
# 2x2 period, these twelve nodes cover all twelve sites exactly once.
KAGOME_STAR_HEXAGON = (
    (-1, 0, 3),
    (-1, 1, 2),
    (0, 1, 1),
    (0, 0, 3),
    (0, 0, 2),
    (0, 0, 1),
)
KAGOME_STAR_APICES = (
    (-1, 1, 1),
    (-1, 1, 3),
    (0, 1, 2),
    (1, 0, 1),
    (0, -1, 3),
    (-1, 0, 2),
)
# Half-width of the star-inset data window; the Kagome bond length is 1.
KAGOME_STAR_RADIUS = 2.18


def kagome_bond_descriptor(
    node0: tuple[int, int, int],
    node1: tuple[int, int, int],
) -> tuple[int, int, str]:
    target = frozenset((node0, node1))
    for source_i in range(-2, 2):
        for source_j in range(-2, 2):
            for bond_name, (start_spec, end_spec) in BOND_ENDPOINTS.items():
                di0, dj0, site0 = start_spec
                di1, dj1, site1 = end_spec
                endpoints = frozenset(
                    (
                        (source_i + di0, source_j + dj0, site0),
                        (source_i + di1, source_j + dj1, site1),
                    )
                )
                if endpoints == target:
                    return source_i, source_j, bond_name
    raise ValueError(f"No kagome J1 bond joins {node0} and {node1}.")


def scientific_notation_tex(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    return rf"{coefficient:.2f}\times10^{{{exponent}}}"


def draw_kagome_observable_pattern(
    ax: plt.Axes,
    unit_cell: str,
    shape: tuple[int, int],
    bonds: list[dict[str, float | int | str]],
    magnetizations: list[dict[str, float | int | str]],
    energy_norm: mcolors.Normalize,
    energy_cmap: mcolors.Colormap,
    moment_reference: float,
) -> None:
    ni, nj = shape
    selected_bonds = [row for row in bonds if row["unit_cell"] == unit_cell]
    selected_mags = [
        row for row in magnetizations if row["unit_cell"] == unit_cell
    ]
    # An inset's spin arrows and bond colors must describe the same state.
    inset_states = {
        (row["D"], row["chi"])
        for row in selected_bonds + selected_mags
    }
    if len(inset_states) != 1:
        raise ValueError(f"Mixed observable sources in the {unit_cell} inset")
    inset_dimension = int(next(iter(inset_states))[0])
    bond_lookup = {
        (int(row["i"]), int(row["j"]), str(row["bond"])): float(row["E"])
        for row in selected_bonds
    }
    mag_lookup = {
        (int(row["i"]), int(row["j"]), int(row["site"])): row
        for row in selected_mags
    }

    def periodic_index(index: int, period: int) -> int:
        return (index - 1) % period + 1

    visible_sites: dict[tuple[int, int, int], np.ndarray] = {}
    for k, node in enumerate(KAGOME_STAR_HEXAGON):
        angle = np.deg2rad(180.0 - 60.0 * k)
        visible_sites[node] = np.array([np.cos(angle), np.sin(angle)])
    for k, node in enumerate(KAGOME_STAR_APICES):
        angle = np.deg2rad(150.0 - 60.0 * k)
        visible_sites[node] = SQRT3 * np.array([np.cos(angle), np.sin(angle)])

    star_edges = []
    for k, apex in enumerate(KAGOME_STAR_APICES):
        hex0 = KAGOME_STAR_HEXAGON[k]
        hex1 = KAGOME_STAR_HEXAGON[(k + 1) % 6]
        star_edges.extend(((hex0, hex1), (hex0, apex), (hex1, apex)))

    for node0, node1 in star_edges:
        source_i, source_j, bond_name = kagome_bond_descriptor(node0, node1)
        pattern_i = periodic_index(source_i, ni)
        pattern_j = periodic_index(source_j, nj)
        energy = bond_lookup[(pattern_i, pattern_j, bond_name)]
        normalized_strength = 1.0 - energy_norm(energy)
        line_width = 0.65 + 1.25 * normalized_strength
        p0 = visible_sites[node0]
        p1 = visible_sites[node1]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=energy_cmap(energy_norm(energy)),
            linewidth=line_width,
            alpha=0.95,
            solid_capstyle="round",
            zorder=1,
        )

    for (i, j, site), point in visible_sites.items():
        ax.scatter(
            [point[0]],
            [point[1]],
            s=8.5,
            facecolor="white",
            edgecolor="0.15",
            linewidth=0.45,
            zorder=3,
        )
        pattern_i = periodic_index(i, ni)
        pattern_j = periodic_index(j, nj)
        row = mag_lookup[(pattern_i, pattern_j, site)]
        vector = np.array([float(row["Mx"]), float(row["Mz"])])
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0.0:
            continue
        relative_magnitude = float(row["Mnorm"]) / moment_reference
        arrow_length = 0.42 * relative_magnitude
        delta = arrow_length * vector / magnitude
        ax.arrow(
            point[0],
            point[1],
            delta[0],
            delta[1],
            width=0.022 * relative_magnitude,
            head_width=0.12 * relative_magnitude,
            head_length=0.14 * relative_magnitude,
            length_includes_head=True,
            color="black",
            linewidth=0.0,
            zorder=4,
        )

    ax.set_xlim(-KAGOME_STAR_RADIUS, KAGOME_STAR_RADIUS)
    ax.set_ylim(-KAGOME_STAR_RADIUS, KAGOME_STAR_RADIUS)
    ax.set_aspect("equal")
    ax.set_facecolor((1.0, 1.0, 1.0, 0.90))
    ax.axis("off")
    mean_magnitude = float(np.mean([float(row["Mnorm"]) for row in selected_mags]))
    title = rf"${ni}\times{nj}$" + "\n" + rf"$D={inset_dimension}$"
    ax.text(
        0.0,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=1.0,
            pad=0.3,
        ),
    )
    ax.text(
        0.5,
        0.02,
        rf"$\overline{{|M|}}={scientific_notation_tex(mean_magnitude)}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=0.3,
        ),
    )


def plot_energy_with_unit_cells(
    kagome: list[dict[str, float | int | str]],
    kagome_energy_fit_points: list[dict[str, float | int | str]],
    two_by_two_fit_points: list[dict[str, float | int | str]],
    energy_benchmarks: list[dict[str, float | int | str]],
    bonds: list[dict[str, float | int | str]],
    magnetizations: list[dict[str, float | int | str]],
    kagome_energy_fit: tuple[float, float, float, float],
    two_by_two_energy_fit: tuple[float, float, float, float],
    two_by_two_energy_error: np.ndarray,
    two_by_two_intercept_error: tuple[float, float],
) -> None:
    fig, ax_energy = plt.subplots(figsize=(3.50, 3.05))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.17, top=0.985)

    energy_x = np.array([1.0 / float(point["xi"]) ** 3 for point in kagome])
    energy_y = np.array([float(point["E"]) for point in kagome])
    selected_point_ids = {id(point) for point in kagome_energy_fit_points}
    excluded_energy_points = [
        point for point in kagome if id(point) not in selected_point_ids
    ]
    selected_energy_x = np.array(
        [1.0 / float(point["xi"]) ** 3 for point in kagome_energy_fit_points]
    )
    selected_energy_y = np.array(
        [float(point["E"]) for point in kagome_energy_fit_points]
    )
    (
        energy_slope,
        energy_intercept,
        energy_intercept_error,
        _,
    ) = kagome_energy_fit
    energy_fit_x = np.linspace(0.0, float(energy_x.max()) * 1.04, 240)
    two_by_two_fit_x_points = np.array(
        [1.0 / float(point["xi"]) ** 3 for point in two_by_two_fit_points]
    )
    two_by_two_fit_y_points = np.array(
        [float(point["E"]) for point in two_by_two_fit_points]
    )
    two_by_two_slope, two_by_two_intercept, _, _ = two_by_two_energy_fit
    two_by_two_fit_x = np.linspace(
        0.0, float(two_by_two_fit_x_points.max()) * 1.04, 240
    )
    if excluded_energy_points:
        ax_energy.scatter(
            [1.0 / float(point["xi"]) ** 3 for point in excluded_energy_points],
            [float(point["E"]) for point in excluded_energy_points],
            facecolor="white",
            edgecolor=VERMILLION,
            marker="s",
            s=17,
            linewidth=0.75,
            alpha=0.82,
            label="_nolegend_",
            zorder=3,
        )
    one_by_one_data_artist = ax_energy.scatter(
        selected_energy_x,
        selected_energy_y,
        color=VERMILLION,
        marker="s",
        s=17,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.92,
        label=r"$1\times1$ lowest-energy data",
        zorder=4,
    )
    (one_by_one_fit_artist,) = ax_energy.plot(
        energy_fit_x,
        energy_slope * energy_fit_x + energy_intercept,
        color=VERMILLION,
        linestyle="-.",
        linewidth=0.85,
        label="_nolegend_",
        zorder=2,
    )
    ax_energy.errorbar(
        [0.0],
        [energy_intercept],
        yerr=[[energy_intercept_error], [energy_intercept_error]],
        fmt="*",
        color=VERMILLION,
        markeredgecolor="black",
        markeredgewidth=0.45,
        markersize=7.2,
        ecolor=VERMILLION,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        label="_nolegend_",
        clip_on=False,
        zorder=6,
    )
    two_by_two_data_artist = ax_energy.scatter(
        two_by_two_fit_x_points,
        two_by_two_fit_y_points,
        marker="^",
        s=4.6**2,
        facecolor=BLUISH_GREEN,
        edgecolor="black",
        linewidth=0.40,
        alpha=0.95,
        label=r"$2\times2$ large unit",
        zorder=4,
    )
    (two_by_two_fit_artist,) = ax_energy.plot(
        two_by_two_fit_x,
        two_by_two_slope * two_by_two_fit_x + two_by_two_intercept,
        color=BLUISH_GREEN,
        linestyle="-.",
        linewidth=0.95,
        label="_nolegend_",
        zorder=2,
    )
    ax_energy.errorbar(
        [0.0],
        [two_by_two_intercept],
        yerr=[
            [two_by_two_intercept_error[0]],
            [two_by_two_intercept_error[1]],
        ],
        fmt="*",
        color=BLUISH_GREEN,
        markeredgecolor="black",
        markeredgewidth=0.45,
        markersize=7.2,
        ecolor=BLUISH_GREEN,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        label="_nolegend_",
        clip_on=False,
        zorder=6,
    )

    benchmark_colors = {"iPEPS AD": VERMILLION, "3-PESS SU": BLUE}
    plot_x_max = max(
        float(energy_fit_x.max()), float(two_by_two_fit_x.max())
    )
    for benchmark in energy_benchmarks:
        method = str(benchmark["method"])
        bond_dimension = int(benchmark["D"])
        energy = float(benchmark["E"])
        color = benchmark_colors[method]
        if method == "3-PESS SU" and bond_dimension == 16:
            ax_energy.plot(
                [0.0, 0.64 * plot_x_max],
                [energy, energy],
                color=color,
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
        else:
            ax_energy.axhline(
                energy,
                color=color,
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
        if method == "iPEPS AD":
            label_x, label_y, ha, va = (
                0.98 * plot_x_max,
                energy - 0.000035,
                "right",
                "top",
            )
        elif bond_dimension == 16:
            label_x, label_y, ha, va = (
                0.015 * plot_x_max,
                energy - 0.000035,
                "left",
                "top",
            )
        else:
            label_x, label_y, ha, va = (
                0.015 * plot_x_max,
                energy - 0.000035,
                "left",
                "top",
            )
        ax_energy.text(
            label_x,
            label_y,
            rf"{method}, $D={bond_dimension}$",
            color="black",
            fontsize=5.7,
            ha=ha,
            va=va,
        )

    ax_energy.set_xlabel(r"$1/\xi^3$")
    ax_energy.set_ylabel(r"$E$ (energy per site)")
    ax_energy.set_xlim(0.0, plot_x_max)
    ax_energy.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_energy.legend(
        handles=[
            one_by_one_data_artist,
            two_by_two_data_artist,
        ],
        labels=[
            r"$1\times1$ single unit",
            r"$2\times2$ large unit",
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.71),
        fontsize=4.9,
        borderaxespad=0.3,
        handlelength=1.35,
        labelspacing=0.25,
    )
    ax_energy.tick_params(width=0.55, length=2.5, direction="out")

    bond_values = np.array([float(row["E"]) for row in bonds])
    energy_norm = mcolors.Normalize(
        vmin=float(bond_values.min()), vmax=float(bond_values.max())
    )
    energy_cmap = plt.get_cmap("cividis_r")
    moment_reference = max(float(row["Mnorm"]) for row in magnetizations)
    star_inset_size = (0.35, 0.32)
    ax_one_site = ax_energy.inset_axes([0.015, 0.620, *star_inset_size])
    ax_two_by_two = ax_energy.inset_axes([0.635, 0.010, *star_inset_size])
    draw_kagome_observable_pattern(
        ax_one_site,
        "1x1x3",
        (1, 1),
        bonds,
        magnetizations,
        energy_norm,
        energy_cmap,
        moment_reference,
    )
    draw_kagome_observable_pattern(
        ax_two_by_two,
        "2x2",
        (2, 2),
        bonds,
        magnetizations,
        energy_norm,
        energy_cmap,
        moment_reference,
    )
    colorbar_axis = ax_energy.inset_axes([0.36, 0.055, 0.19, 0.018])
    scalar_mappable = cm.ScalarMappable(norm=energy_norm, cmap=energy_cmap)
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.solids.set_rasterized(False)
    colorbar.ax.set_title(
        r"$\langle\mathbf{S}_i\!\cdot\!\mathbf{S}_j\rangle$",
        fontsize=5.2,
        pad=1.0,
    )
    colorbar.set_ticks([energy_norm.vmin, energy_norm.vmax])
    colorbar.ax.tick_params(labelsize=5.2, width=0.45, length=1.8, pad=1.0)
    colorbar.outline.set_linewidth(0.5)

    # The iPEPS ansatz (three Kagome sites blocked into one square-lattice
    # tensor) fills the empty upper-middle region of the energy panel.  Both
    # drawings use a unit Kagome bond length, so sizing the inset with the
    # star insets' inches per data unit gives the same lattice spacing.
    axes_box = ax_energy.get_position()
    axes_width_in = axes_box.width * fig.get_figwidth()
    axes_height_in = axes_box.height * fig.get_figheight()
    inches_per_bond = min(
        star_inset_size[0] * axes_width_in,
        star_inset_size[1] * axes_height_in,
    ) / (2.0 * KAGOME_STAR_RADIUS)
    ansatz_xlim = (-1.95, 2.75)
    ansatz_ylim = (-0.45, 2.85)
    ansatz_width = (
        (ansatz_xlim[1] - ansatz_xlim[0]) * inches_per_bond / axes_width_in
    )
    ansatz_height = (
        (ansatz_ylim[1] - ansatz_ylim[0]) * inches_per_bond / axes_height_in
    )
    ansatz_center = (0.545, 0.620 + 0.5 * star_inset_size[1])
    ax_ansatz = ax_energy.inset_axes(
        [
            ansatz_center[0] - 0.5 * ansatz_width,
            ansatz_center[1] - 0.5 * ansatz_height,
            ansatz_width,
            ansatz_height,
        ]
    )
    draw_kagome_ansatz(
        ax_ansatz, lw_scale=0.33, xlim=ansatz_xlim, ylim=ansatz_ylim
    )

    fig.savefig(ENERGY_OUT_STEM.with_suffix(".pdf"))
    plt.close(fig)


def plot_magnetic_scaling(
    square: list[dict[str, float | int | str]],
    kagome: list[dict[str, float | int | str]],
    kagome_fit_points: list[dict[str, float | int | str]],
    square_fit: tuple[float, float, float],
    kagome_fit: tuple[float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.55))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.97)
    styles = {
        "Square": {
            "points": square,
            "color": BLUE,
            "marker": "o",
            "linestyle": "-.",
            "fit": square_fit,
            "fit_points": square,
            "line_points": square,
            "legend_label": rf"Square, $\alpha={square_fit[0]:.3f}$",
        },
        "Kagome": {
            "points": kagome,
            "color": VERMILLION,
            "marker": "s",
            "linestyle": "-.",
            "fit": kagome_fit,
            "fit_points": kagome_fit_points,
            "line_points": kagome,
            "legend_label": rf"Kagome, $\alpha={kagome_fit[0]:.3f}$",
        },
    }
    for label, style in styles.items():
        points = style["points"]
        fit_points = style["fit_points"]
        fit_point_ids = {id(point) for point in fit_points}
        excluded_points = [
            point for point in points if id(point) not in fit_point_ids
        ]
        if excluded_points:
            ax.scatter(
                [float(point["log_inv_xi"]) for point in excluded_points],
                [float(point["log_M2"]) for point in excluded_points],
                facecolor="white",
                edgecolor=style["color"],
                marker=style["marker"],
                s=17,
                linewidth=0.75,
                alpha=0.92,
                zorder=3,
            )
        ax.scatter(
            [float(point["log_inv_xi"]) for point in fit_points],
            [float(point["log_M2"]) for point in fit_points],
            color=style["color"],
            marker=style["marker"],
            s=17,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.92,
            label=style["legend_label"],
            zorder=3,
        )
        slope, intercept, _ = style["fit"]
        fit_x = np.array(
            [float(point["log_inv_xi"]) for point in style["line_points"]]
        )
        xfit = np.linspace(float(fit_x.min()), float(fit_x.max()), 240)
        ax.plot(
            xfit,
            slope * xfit + intercept,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.85,
            zorder=2,
        )

    # Smallest measured 3-PESS magnetization; this reference has no xi and is
    # not part of either scaling fit.
    with PESS_MAGNETIZATION_CSV.open(newline="", encoding="utf-8") as handle:
        pess_m = float(next(csv.DictReader(handle))["M"])
    pess_log_m2 = float(np.log(pess_m**2))
    ax.axhline(
        pess_log_m2,
        color=VERMILLION,
        linestyle="--",
        linewidth=0.8,
        label=r"Kagome 3-PESS SU, minimum $m$",
        zorder=1,
    )

    ax.set_xlabel(r"$\log(1/\xi)$")
    ax.set_ylabel(r"$\log(m^2)$")
    ax.legend(frameon=False, loc="center", fontsize=5.8)
    ax.tick_params(width=0.55, length=2.5, direction="out")
    fig.savefig(SCALING_OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot() -> None:
    grouped = load_points()
    square = grouped["Square"]
    kagome = grouped["Kagome"]
    energy_benchmarks = load_energy_benchmarks()
    two_by_two = load_optimized_two_by_two_energies()
    two_by_two_fit_points = select_two_by_two_max_chi_points(two_by_two)
    kagome_energy_fit_points = select_lowest_energy_points(kagome, 4)
    kagome_magnetic_fit_points = select_lowest_magnetization_points(kagome, 4)
    bonds = load_bonds()
    magnetizations = load_magnetizations()
    square_fit = fit_line(square)
    kagome_fit = fit_line(kagome_magnetic_fit_points)
    kagome_energy_fit = fit_kagome_energy(kagome_energy_fit_points)
    two_by_two_energy_fit = fit_kagome_energy(two_by_two_fit_points)
    (
        two_by_two_energy_error,
        two_by_two_intercept_error,
    ) = estimate_two_by_two_finite_chi_sensitivity(
        two_by_two,
        two_by_two_fit_points,
        two_by_two_energy_fit[1],
    )

    plot_energy_with_unit_cells(
        kagome,
        kagome_energy_fit_points,
        two_by_two_fit_points,
        energy_benchmarks,
        bonds,
        magnetizations,
        kagome_energy_fit,
        two_by_two_energy_fit,
        two_by_two_energy_error,
        two_by_two_intercept_error,
    )
    plot_magnetic_scaling(
        square,
        kagome,
        kagome_magnetic_fit_points,
        square_fit,
        kagome_fit,
    )

    print(
        f"Square points: {len(square)}; slope={square_fit[0]:.8f}; "
        f"R2={square_fit[2]:.8f}"
    )
    magnetic_label = ",".join(
        f"D={point['D']}" for point in sorted(
            kagome_magnetic_fit_points, key=lambda point: int(point["D"])
        )
    )
    print(
        f"Kagome {magnetic_label} magnetic points: "
        f"{len(kagome_magnetic_fit_points)}; "
        f"slope={kagome_fit[0]:.8f}; "
        f"R2={kagome_fit[2]:.8f}"
    )
    for point in kagome_magnetic_fit_points:
        print(
            f"Kagome magnetic fit point: D={point['D']}; "
            f"chi={point['chi']}; log_M2={point['log_M2']}; xi={point['xi']}"
        )
    energy_label = ",".join(
        f"D={point['D']}" for point in sorted(
            kagome_energy_fit_points, key=lambda point: int(point["D"])
        )
    )
    print(
        f"Kagome {energy_label} energy fit: "
        f"E_inf={kagome_energy_fit[1]:.10f} +/- "
        f"{kagome_energy_fit[2]:.10f}; R2={kagome_energy_fit[3]:.8f}"
    )
    for point in kagome_energy_fit_points:
        print(
            f"Kagome 1x1 fit point: D={point['D']}; "
            f"chi={point['chi']}; E={point['E']}; xi={point['xi']}"
        )
    print(
        "Kagome 2x2 max-chi three-point extrapolation: "
        f"E_inf={two_by_two_energy_fit[1]:.10f} "
        f"+/-{two_by_two_energy_fit[2]:.10f}; "
        f"R2={two_by_two_energy_fit[3]:.8f}; finite-chi envelope "
        f"-{two_by_two_intercept_error[0]:.10f} "
        f"+{two_by_two_intercept_error[1]:.10f}"
    )
    for point in two_by_two_fit_points:
        print(
            f"Kagome 2x2 fit point: D={point['D']}; "
            f"chi={point['chi']}; E={point['E']}; xi={point['xi']}"
        )
    for point in two_by_two:
        print(
            f"Kagome 2x2 optimized point: D={point['D']}; "
            f"chi={point['chi']}; E={point['E']}; xi={point['xi']}"
        )
    print(f"saved {ENERGY_OUT_STEM.with_suffix('.pdf')}")
    print(f"saved {SCALING_OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    plot()
