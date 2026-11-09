import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_STEM = SCRIPT_DIR / "J05_E_extrap_xi3"
MM_TO_IN = 1 / 25.4

BLUE = "#0072B2"
VERMILLION = "#D55E00"
BLUISH_GREEN = "#009E73"
ORANGE = "#E69F00"
REDDISH_PURPLE = "#CC79A7"


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.2,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
})


def load_ipeps_points(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "D": int(row["D"]),
                "chi": int(row["chi"]),
                "E": float(row["E"]),
                "inv_xi3": float(row["inv_xi3"]),
            })
    return rows


def load_reference_points(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "label": row["label"],
                "kind": row["kind"],
                "L": None if row["L"] == "" else float(row["L"]),
                "E": float(row["E"]),
            })
    return rows


def linear_fit(x, y):
    amat = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(amat, y, rcond=None)
    yhat = intercept + slope * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


ipeps = load_ipeps_points(DATA_DIR / "J05_E_extrap_ipeps.csv")
refs = load_reference_points(DATA_DIR / "J05_E_extrap_references.csv")

x = np.array([p["inv_xi3"] for p in ipeps])
y = np.array([p["E"] for p in ipeps])
slope, einfr, r2 = linear_fit(x, y)

finite_size = [r for r in refs if r["kind"] == "finite_size"]
x_fs = np.array([1.0 / r["L"] ** 3 for r in finite_size])
y_fs = np.array([r["E"] for r in finite_size])
slope_fs, einf_fs, r2_fs = linear_fit(x_fs, y_fs)

fig, ax = plt.subplots(figsize=(60 * MM_TO_IN, 60 * MM_TO_IN))
fig.subplots_adjust(left=0.20, right=0.90, bottom=0.15, top=0.85)
fL = 0.04

ax.scatter(x, y, color=BLUE, s=13, edgecolor="black", linewidth=0.35, zorder=3,
           label="this work")

xf = np.linspace(0, x.max() * 1.05, 50)
ax.plot(xf, einfr + slope * xf, color=BLUE, ls="--", lw=0.85,
        label=r"iPEPS fit")
ax.scatter([0], [einfr], marker="*", s=44, color=BLUE, ec="k", lw=0.45,
           zorder=6)

xr = x.max() * 1.05
ax.set_xlim(-fL * xr, xr)
ax.set_xlabel(r"$1/\xi^3$")
ax.set_ylabel(r"$E$ per site")
ax.set_ylim(-0.49775, -0.49643)

ax2 = ax.twiny()
ax2.scatter(x_fs, y_fs, color="black", marker="s", s=15, zorder=3,
            label="CNN-MPS")

xff = np.linspace(0, x_fs.max() * 1.15, 50)
ax2.plot(xff, einf_fs + slope_fs * xff, color="black", ls=":", lw=0.85,
         label="_nolegend_")

point_styles = {
    "Res.+Lanc.": {"color": ORANGE, "marker": "P", "s": 32},
    "ViT": {"color": VERMILLION, "marker": "^", "s": 34},
    "PBC PEPS": {"color": REDDISH_PURPLE, "marker": "D", "s": 28},
    "OBC PEPS": {"color": BLUISH_GREEN, "marker": "v", "s": 34},
}
seen_point_labels = set()
for ref in refs:
    if ref["kind"] != "point":
        continue
    style = point_styles[ref["label"]]
    label = ref["label"] if ref["label"] not in seen_point_labels else "_nolegend_"
    seen_point_labels.add(ref["label"])
    ax2.scatter([1.0 / ref["L"] ** 3], [ref["E"]], zorder=5, ec="k", lw=0.5,
                label=label, **style)

for ref in refs:
    if ref["kind"] == "hline" and ref["label"] == "QRCTM":
        ax.axhline(ref["E"], color=BLUISH_GREEN, ls="-.", lw=0.85,
                   label=r"$C_{4v}$ iPEPS")

xr2 = x_fs.max() * 1.15
ax2.set_xlim(-fL * xr2, xr2)
ax2.set_xlabel(r"$1/L^3$")

# Keep the primary iPEPS scale at the bottom and the finite-size literature
# scale at the top, following the usual twiny convention.
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()
ax.tick_params(axis="x", colors="black", top=False, bottom=True,
               labeltop=False, labelbottom=True, direction="out")
ax2.xaxis.set_label_position("top")
ax2.xaxis.tick_top()
ax2.tick_params(axis="x", colors="black", top=True, bottom=False,
                labeltop=True, labelbottom=False, direction="out")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2,
          columnspacing=0.5, handletextpad=0.3, handlelength=1.3,
          borderpad=0.1, frameon=False)

ax.text(-0.115, 1.02, "c", transform=ax.transAxes,
        fontsize=8.2, fontweight="bold", va="bottom", ha="left")

fig.savefig(OUT_STEM.with_suffix(".pdf"))

print(f"iPEPS fit: E_inf={einfr:.12f}, slope={slope:.12f}, R^2={r2:.6f}")
print(f"CNN-MPS finite-size fit: E_inf={einf_fs:.12f}, "
      f"slope={slope_fs:.12f}, R^2={r2_fs:.6f}")
print("saved", OUT_STEM.with_suffix(".pdf"))
