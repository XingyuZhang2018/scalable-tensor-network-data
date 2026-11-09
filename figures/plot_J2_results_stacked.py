import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

MM_TO_IN = 1 / 25.4
FIGSIZE = (176 * MM_TO_IN, 88 * MM_TO_IN)

NEEL_COLOR = (0.0, 0.447, 0.698)
PLAQ_COLOR = (0.835, 0.369, 0.0)
HIGHLIGHT_J2 = 0.50
INSET_LABEL_SIZE = 9.5
HIGHLIGHT_LABEL_SIZE = 9.5
PANEL_LABEL_SIZE = 12.0
NEEL_ARROW_LINEWIDTH = 1.65
PLAQUETTE_LINEWIDTH = 1.7
HIGHLIGHT_LINEWIDTH = 1.05


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.5,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "mathtext.fontset": "stix",
})


def load_fit_points(path):
    fit_points = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            j2 = float(row["j2"])
            fit_points[j2].append({
                "D": int(row["D"]),
                "chi": int(row["chi"]),
                "M2": float(row["M2"]),
                "inv_xi_site": float(row["inv_xi_site"]),
                "kept_fit": bool(int(row["kept_fit"])),
            })
    return dict(fit_points)


def load_fit_results(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "j2": float(row["j2"]),
                "M2_ext": float(row["M2_ext"]),
                "M_ext": float(row["M_ext"]),
                "slope": float(row["slope"]),
            })
    return sorted(rows, key=lambda r: r["j2"])


def load_vbs_order(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "j2": float(row["j2"]),
                "P": float(row["P"]),
                "D": int(row["D"]),
                "chi": int(row["chi"]),
            })
    return sorted(rows, key=lambda r: r["j2"])


def style_axes(ax):
    ax.grid(False, which="both", axis="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.55, length=3.0, pad=2)


def draw_neel_inset(parent):
    iax = parent.inset_axes([0.010, 0.12, 0.285, 0.285])
    iax.set_aspect("equal")
    iax.set_xlim(-0.35, 3.35)
    iax.set_ylim(-0.52, 3.35)
    iax.axis("off")

    for i in range(4):
        iax.plot([i, i], [0, 3], color="black", lw=0.55, alpha=0.75, zorder=1)
        iax.plot([0, 3], [i, i], color="black", lw=0.55, alpha=0.75, zorder=1)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    iax.scatter(xs.ravel(), ys.ravel(), s=58, c="#c9c9c9", alpha=0.76,
                edgecolors="none", zorder=2)
    for i in range(4):
        for j in range(4):
            up = (i + j) % 2 == 0
            color = "#0b4f8a" if up else "#d7301f"
            sign = 1.0 if up else -1.0
            half_arrow = 0.20
            arrow = FancyArrowPatch(
                (i - sign * half_arrow, j - sign * half_arrow),
                (i + sign * half_arrow, j + sign * half_arrow),
                arrowstyle="-|>",
                mutation_scale=7.5,
                lw=NEEL_ARROW_LINEWIDTH,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
            iax.add_patch(arrow)
    iax.text(1.5, -0.42, "Néel", ha="center", va="top",
             fontsize=INSET_LABEL_SIZE, color="0.25")


def draw_plaquette_inset(parent):
    iax = parent.inset_axes([0.675, 0.12, 0.285, 0.285])
    iax.set_aspect("equal")
    iax.set_xlim(-0.35, 3.35)
    iax.set_ylim(-0.52, 3.35)
    iax.axis("off")

    for i in range(4):
        iax.plot([i, i], [0, 3], color="black", lw=0.45, alpha=0.36, zorder=1)
        iax.plot([0, 3], [i, i], color="black", lw=0.45, alpha=0.36, zorder=1)
    for i0 in (0, 2):
        for j0 in (0, 2):
            xs = [i0, i0 + 1, i0 + 1, i0, i0]
            ys = [j0, j0, j0 + 1, j0 + 1, j0]
            iax.plot(xs, ys, color=PLAQ_COLOR,
                     lw=PLAQUETTE_LINEWIDTH, zorder=3)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    iax.scatter(xs.ravel(), ys.ravel(), s=8, c="black", zorder=4)
    iax.text(1.5, -0.42, "Plaquette", ha="center", va="top",
             fontsize=INSET_LABEL_SIZE, color="0.25")


def main(output_stem="J2_sweep_extrap", figsize=FIGSIZE):
    fit_points = load_fit_points(DATA_DIR / "J2_sweep_fit_points.csv")
    fit_results = load_fit_results(DATA_DIR / "J2_sweep_fit_results.csv")
    vbs_order = load_vbs_order(DATA_DIR / "J2_sweep_vbs_order.csv")

    j2_values = sorted(fit_points)
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=min(j2_values), vmax=max(j2_values))
    color_for = {j: cmap(norm(j)) for j in j2_values}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        left=0.090,
        right=0.900,
        bottom=0.19,
        top=0.92,
        wspace=0.26,
    )

    ax_order = fig.add_subplot(gs[0, 0])
    ax_m2 = fig.add_subplot(gs[0, 1])
    cax = make_axes_locatable(ax_m2).append_axes("right", size="4.0%", pad=0.035)

    js = [r["j2"] for r in fit_results]
    m_ext = [r["M_ext"] for r in fit_results]
    ax_order.plot(js, m_ext, color=NEEL_COLOR, marker="*", ms=4.2, lw=0.85,
                  label=r"$m_{\mathrm{extr.}} = \sqrt{(m^2)_{\mathrm{extr.}}}$")

    js_p = [r["j2"] for r in vbs_order]
    ps = [r["P"] for r in vbs_order]
    ax_order.plot(js_p, ps, color=PLAQ_COLOR, marker="^", ms=2.8, lw=0.85,
                  label=(
                      r"$(E_{\langle ij\rangle,\mathrm{max}}"
                      r"-E_{\langle ij\rangle,\mathrm{min}})/J_1$"
                  ))

    ax_order.axhline(0, color="0.35", lw=0.55, ls=":")
    ax_order.axvspan(0.485, 0.515, color="0.5", alpha=0.08, zorder=0)
    ax_order.set_xlabel(r"$J_2/J_1$")
    ax_order.set_ylabel("Order parameters")
    ax_order.set_xlim(0.39, 0.61)
    ax_order.set_ylim(-0.012, 0.19)
    ax_order.xaxis.set_major_locator(mticker.FixedLocator([0.40, 0.45, 0.50, 0.55, 0.60]))
    ax_order.yaxis.set_major_locator(mticker.FixedLocator([0.00, 0.05, 0.10, 0.15]))
    ax_order.legend(loc="upper right", frameon=False, handlelength=1.4, borderpad=0.1)
    style_axes(ax_order)
    draw_neel_inset(ax_order)
    draw_plaquette_inset(ax_order)

    for r in fit_results:
        pts = fit_points[r["j2"]]
        inv_xi = np.array([p["inv_xi_site"] for p in pts])
        m2 = np.array([p["M2"] for p in pts])
        kept = np.array([p["kept_fit"] for p in pts], dtype=bool)
        col = color_for[r["j2"]]
        ax_m2.scatter(inv_xi[kept], m2[kept], color=col, s=4.4, linewidths=0, zorder=3)
        # Show the raw points without a fit or extrapolated endpoint at J2=0.52.
        if abs(r["j2"] - 0.52) < 1e-10:
            continue
        if kept.any():
            xfit = np.linspace(0, float(inv_xi[kept].max()) * 1.05, 50)
            yfit = r["M2_ext"] + r["slope"] * xfit
            ax_m2.plot(xfit, yfit, color=col, ls=(0, (3, 2)), lw=0.50, alpha=0.75)
            if abs(r["j2"] - HIGHLIGHT_J2) < 1e-10:
                ax_m2.plot(
                    xfit, yfit,
                    color="black",
                    ls=(0, (3.2, 2.0)),
                    lw=HIGHLIGHT_LINEWIDTH,
                    alpha=0.90,
                    zorder=4,
                )
                kept_x = inv_xi[kept]
                kept_y = m2[kept]
                label_idx = int(np.argmax(kept_y))
                x_label = 0.485
                y_label = float(kept_y[label_idx])
                ax_m2.text(
                    x_label,
                    y_label,
                    r"$J_2/J_1=0.50$",
                    ha="right",
                    va="center",
                    fontsize=HIGHLIGHT_LABEL_SIZE,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
                    zorder=6,
                )
        ax_m2.scatter([0.0], [r["M2_ext"]], color=col, marker="*", s=19, zorder=5)

    ax_m2.set_xlabel(r"$1/\xi(D,\chi)$")
    ax_m2.set_ylabel(r"$m^2(D,\chi)$")
    ax_m2.set_xlim(0.0, 0.50)
    ax_m2.set_ylim(-0.004, 0.029)
    ax_m2.yaxis.set_major_locator(mticker.FixedLocator([0.00, 0.01, 0.02]))
    style_axes(ax_m2)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$J_2/J_1$", rotation=90, labelpad=2)
    cb.set_ticks([0.40, 0.50, 0.60])
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    cb.ax.invert_yaxis()
    cb.outline.set_linewidth(0.6)
    cax.tick_params(width=0.55, length=3.0, pad=2)

    ax_order.text(-0.115, 1.02, "a", transform=ax_order.transAxes,
                  fontsize=PANEL_LABEL_SIZE, fontweight="bold",
                  va="bottom", ha="left")
    ax_m2.text(-0.115, 1.02, "b", transform=ax_m2.transAxes,
               fontsize=PANEL_LABEL_SIZE, fontweight="bold",
               va="bottom", ha="left")

    pdf = SCRIPT_DIR / f"{output_stem}.pdf"
    png = SCRIPT_DIR / f"{output_stem}.png"
    fig.savefig(pdf, transparent=False)
    fig.savefig(png, dpi=600, transparent=False)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
