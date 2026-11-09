"""Plot restoration of H/V bond equality in the less-restricted plaquette ansatz.

The tabulated values are bond energies of the D=10 state at J2/J1=0.55.
Each horizontal value is the mean of its two exactly repeated entries; each
vertical value is the mean of the two symmetry-related entries in the same
strong- or weak-bond class.  Only the final high-chi continuation branch is
shown; the figure is a symmetry diagnostic, not a chi extrapolation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "Square_J055_D10_unrestricted_bond_isotropization.csv"
OUTPUT = HERE / "Square_J055_D10_unrestricted_bond_isotropization.pdf"


def main() -> None:
    data = np.genfromtxt(DATA, delimiter=",", names=True)
    chi = data["chi"]
    strong_h = data["strong_horizontal"]
    strong_v = data["strong_vertical"]
    weak_h = data["weak_horizontal"]
    weak_v = data["weak_vertical"]

    delta_strong = np.abs(strong_h - strong_v)
    delta_weak = np.abs(weak_h - weak_v)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.bfit": "Times New Roman:italic:bold",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 4.35),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.10},
    )
    ax, ax_delta = axes

    strong_color = "#2166ac"
    weak_color = "#b2182b"

    ax.plot(chi, strong_h, "o-", color=strong_color, label=r"$E_s^H$")
    ax.plot(
        chi,
        strong_v,
        "s--",
        color=strong_color,
        markerfacecolor="white",
        label=r"$E_s^V$",
    )
    ax.plot(chi, weak_h, "o-", color=weak_color, label=r"$E_w^H$")
    ax.plot(
        chi,
        weak_v,
        "s--",
        color=weak_color,
        markerfacecolor="white",
        label=r"$E_w^V$",
    )
    ax.set_ylabel(r"nearest-neighbor bond energy")
    ax.set_ylim(-0.395, -0.195)
    ax.grid(False, which="both", axis="both")
    ax.legend(loc="center right", ncol=2, columnspacing=0.8, handlelength=2.0)
    ax.text(0.025, 0.020, "a", transform=ax.transAxes, va="bottom", fontweight="bold",
            fontfamily="Arial")

    ax_delta.semilogy(
        chi,
        delta_strong,
        "o-",
        color=strong_color,
        label=r"$|E_{s}^{H}-E_{s}^{V}|$",
    )
    ax_delta.semilogy(
        chi,
        delta_weak,
        "s-",
        color=weak_color,
        label=r"$|E_{w}^{H}-E_{w}^{V}|$",
    )
    ax_delta.set_xlabel(r"environment bond dimension $\chi$")
    ax_delta.set_ylabel(r"directional splitting")
    ax_delta.set_xticks(chi.astype(int))
    ax_delta.set_ylim(1.8e-3, 6.0e-2)
    ax_delta.grid(False, which="both", axis="both")
    ax_delta.legend(loc="upper right")
    ax_delta.text(
        0.025, 0.07, "b", transform=ax_delta.transAxes, va="bottom", fontweight="bold",
        fontfamily="Arial",
    )

    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.12, top=0.985)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {OUTPUT}")
    for c, ds, dw in zip(chi.astype(int), delta_strong, delta_weak):
        print(f"chi={c}: delta_strong={ds:.12g}, delta_weak={dw:.12g}")


if __name__ == "__main__":
    main()
