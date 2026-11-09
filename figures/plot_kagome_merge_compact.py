"""Compact Kagome-to-square blocking schematic inspired by Fig. 2(b).

The three sites A, B, C enclosed by each rounded blue tensor are merged
into one effective square-lattice site.  They deliberately form an L-shaped
Kagome path rather than one elementary triangle.

Reference: Corboz et al., Phys. Rev. B 86, 041106(R) (2012),
https://doi.org/10.1103/PhysRevB.86.041106
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch


# Functional palette shared with Figure 2(a).
TENSOR = "#0B5AA4"
PEPS_BOND = TENSOR
TENSOR_LIGHT = "#86C2DE"
TENSOR_FILL = "#ECF2F8"
PHYSICAL_SITE = "#B34FC8"
KAGOME_BOND = "#9AA2A9"
WHITE = "#FFFFFF"


def project(x: float, y: float) -> tuple[float, float]:
    """Map the square bookkeeping grid to an undistorted Kagome frame."""

    # e_x=(1, 0), e_y=(-1/2, sqrt(3)/2): the horizontal, vertical,
    # and diagonal bookkeeping directions become 0, 120, and 60 degrees.
    return x - 0.5 * y, 0.5 * math.sqrt(3.0) * y


def tensor_path(x: float, y: float) -> MplPath:
    """Rounded triangular path enclosing A=(0,0), B=(1,0), C=(0,1)."""

    ox, oy = project(x, y)
    shape = [
        (-0.12, -0.25),
        (-0.43, -0.17),
        (-0.75, 0.55),
        (-0.68, 0.95),
        (-0.63, 1.17),
        (-0.42, 1.22),
        (-0.18, 1.04),
        (0.48, 0.55),
        (1.15, 0.22),
        (1.16, -0.04),
        (1.16, -0.24),
        (0.40, -0.32),
        (-0.12, -0.25),
        (0.0, 0.0),
    ]
    vertices = [(ox + px, oy + py) for px, py in shape]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def tensor_fill(x: float, y: float) -> PathPatch:
    return PathPatch(
        tensor_path(x, y),
        facecolor=TENSOR_FILL,
        edgecolor="none",
        zorder=1,
    )


def tensor_outline(x: float, y: float, lw_scale: float = 1.0) -> PathPatch:
    return PathPatch(
        tensor_path(x, y),
        facecolor="none",
        edgecolor=TENSOR,
        linewidth=1.55 * lw_scale,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )


def physical_leg_tip(site: tuple[float, float]) -> tuple[float, float]:
    """Tip of a short receding physical leg, as in Figure 2(a)."""

    # Tilt farther from the Kagome lattice's 60-degree bond direction so the
    # physical index stays visually separate from the dashed interactions.
    return site[0] + 0.27, site[1] + 0.23


def draw_kagome_bonds(ax: plt.Axes, lw_scale: float = 1.0) -> None:
    """Draw the physical Kagome interactions above the tensor fill."""

    span = (-3.0, 6.0)
    for k in range(-2, 4):
        p0, p1 = project(span[0], 2 * k), project(span[1], 2 * k)
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=KAGOME_BOND,
                lw=2.05 * lw_scale, alpha=0.82, linestyle=(0, (3.2, 2.6)),
                dash_capstyle="round", zorder=2)
        p0, p1 = project(2 * k, span[0]), project(2 * k, span[1])
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=KAGOME_BOND,
                lw=2.05 * lw_scale, alpha=0.82, linestyle=(0, (3.2, 2.6)),
                dash_capstyle="round", zorder=2)
    for offset in range(-5, 6, 2):
        p0 = project(span[0], span[0] + offset)
        p1 = project(span[1], span[1] + offset)
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=KAGOME_BOND,
                lw=2.05 * lw_scale, alpha=0.82, linestyle=(0, (3.2, 2.6)),
                dash_capstyle="round", zorder=2)


def draw_on_axes(
    ax: plt.Axes,
    lw_scale: float = 1.0,
    xlim: tuple[float, float] = (-2.25, 3.15),
    ylim: tuple[float, float] = (-0.45, 2.85),
) -> None:
    """Draw the blocking schematic into ``ax``.

    Line widths are in points, so ``lw_scale`` shrinks them together with the
    data window when the schematic is embedded as a small inset.
    """

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Connections between the composite tensors form the effective square PEPS.
    for x in range(-2, 5, 2):
        for y in range(-2, 7, 2):
            # Extend both ends beneath the neighboring tensor outlines; this
            # prevents a white antialiasing hairline at either attachment.
            p0, p1 = project(x - 1.00, y + 0.52), project(x - 0.27, y + 0.52)
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]),
                    color=PEPS_BOND, lw=2.50 * lw_scale,
                    solid_capstyle="round", zorder=0)
            # In Fig. 2(b) the inter-row PEPS leg leaves the middle of the
            # tensor's lower edge (halfway between A and B), not the B corner.
            p0, p1 = project(x + 0.50, y + 1.18), project(x + 0.50, y + 1.74)
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]),
                    color=PEPS_BOND, lw=2.50 * lw_scale,
                    solid_capstyle="round", zorder=0)

    # Each Kagome site carries its own physical leg.  A common, nearly
    # vertical offset makes the legs read as receding from the lattice plane.
    sites = []
    for x in range(-3, 6):
        for y in range(-2, 6):
            if x % 2 == 0 or y % 2 == 0:
                site = project(x, y)
                tip = physical_leg_tip(site)
                ax.plot((site[0], tip[0]), (site[1], tip[1]),
                        color=PHYSICAL_SITE, lw=1.80 * lw_scale,
                        solid_capstyle="round", zorder=2.5)
                sites.append((site, tip))

    # Opaque fills clip the PEPS bonds exactly at the tensor boundaries.
    for x in range(-2, 5, 2):
        for y in range(-2, 7, 2):
            ax.add_patch(tensor_fill(x, y))

    # Kagome interactions remain visible inside the blocked tensors.
    draw_kagome_bonds(ax, lw_scale)

    # Repeating tensor outlines and physical sites.
    for x in range(-2, 5, 2):
        for y in range(-2, 7, 2):
            ax.add_patch(tensor_outline(x, y, lw_scale))
    for site, _tip in sites:
        ax.add_patch(Circle(site, 0.125, facecolor=TENSOR,
                            edgecolor=TENSOR_LIGHT,
                            linewidth=0.65 * lw_scale, zorder=4))


def draw(output: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    draw_on_axes(ax)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.015,
                facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("kagome_merge_compact.png"),
        help="Output image (PNG, PDF, SVG, ...; default: beside this script).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution.")
    args = parser.parse_args()
    draw(args.output, args.dpi)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
