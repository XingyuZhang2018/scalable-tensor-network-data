"""Plot an appendix illustration of finite-chi evaluation and reoptimization."""
from pathlib import Path
import csv
import argparse
import importlib.util
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-stem', type=Path, default=SCRIPT_DIR / 'heisenberg_chi_reoptimization')
args = parser.parse_args()
OUT_STEM = args.output_stem.resolve()
OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
POINT_DATA = DATA_DIR / 'heisenberg_chi_reoptimization.csv'
FIT_SCRIPT = SCRIPT_DIR / 'plot_heisenberg_qmc_ipeps_extrapolation.py'

# Reuse the actual Fig.5 energy-plot function and its plotted curve samples.
# There is no separately implemented fit or fit to the three example points.
spec = importlib.util.spec_from_file_location('heisenberg_fig5', FIT_SCRIPT)
fig5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig5)
reference_figure, reference_axes = plt.subplots()
slope, intercept, r2, intercept_error = fig5.plot_energy(reference_axes)
reference_line = next(line for line in reference_axes.lines if line.get_label() == 'iPEPS fit')
reference_x = np.array(reference_line.get_xdata(), dtype=float, copy=True)
reference_y = np.array(reference_line.get_ydata(), dtype=float, copy=True)
plt.close(reference_figure)
plt.rcdefaults()

with POINT_DATA.open(newline='', encoding='utf-8') as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 3
points = []
for row in raw:
    point = dict(row)
    for key in ['D', 'chi_opt', 'chi_eval']:
        point[key] = int(row[key])
    for key in ['E', 'xi']:
        point[key] = float(row[key])
    point['x'] = 1e4 / point['xi']**3
    point['y'] = point['E']
    point['fit_residual'] = point['E'] - (intercept + slope / point['xi']**3)
    points.append(point)

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'stix', 'font.size': 7,
                     'axes.labelsize': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': True, 'axes.spines.right': True,
                     'axes.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(89/25.4, 72/25.4))
fig.subplots_adjust(left=.28, right=.985, bottom=.20, top=.97)
xs = np.array([p['x'] for p in points])
ys = np.array([p['y'] for p in points])
xlim = (1.65, 2.85)
xx = reference_x * 1e4
yy = reference_y
black, green, red = '#252a32', '#178449', '#bf4235'
ax.plot(xx, yy, color=black, lw=1.05, ls='--', label='iPEPS fit')
for i, p in enumerate(points):
    ax.scatter(p['x'], p['y'], s=22, facecolors='white' if i == 1 else black,
               edgecolors=green if i == 1 else black, linewidths=0.85, zorder=5)
for i, color in [(0, green), (1, red)]:
    a, b = points[i:i+2]
    ax.annotate('', xy=(b['x'], b['y']), xytext=(a['x'], a['y']),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color=color,
                                mutation_scale=8, shrinkA=3.5, shrinkB=3.5))
labels = [
    (r'optimized $\chi=1024$', (-5, 7), 'right', black),
    (r'evaluated $\chi=1536$', (4, -9), 'left', green),
    (r'optimized $\chi=1536$', (5, -13), 'left', black),
]
for p, (label, offset, ha, color) in zip(points, labels):
    ax.annotate(label, (p['x'], p['y']), xytext=offset, textcoords='offset points',
                ha=ha, fontsize=6.5, color=color)
for i, text, color, offset in [(0, 'fixed tensor', green, (5, 0)),
                               (1, 'reoptimize', red, (0, 8))]:
    a, b = points[i:i+2]
    ax.annotate(text, ((a['x']+b['x'])/2, (a['y']+b['y'])/2),
                xytext=offset, textcoords='offset points',
                ha='left' if i == 0 else 'center', va='center',
                color=color, fontsize=6.5)
yrange = max(float(np.ptp(ys)), 1e-9)
ax.set(xlim=xlim,
       ylim=(ys.min() - .24*yrange, ys.max() + .28*yrange),
       xlabel=r'$1/\xi^3\;(\times10^{-4})$', ylabel=r'$E/J_1$')
ax.yaxis.set_major_locator(MultipleLocator(1e-7))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.7f'))
ax.grid(False, which="both", axis="both")
ax.legend(frameon=False, loc='upper left', fontsize=6.2)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT_STEM.with_suffix('.'+ext), dpi=300, facecolor='white')

print(json.dumps({'slope': slope, 'intercept': intercept,
                  'reference_artist_samples': len(reference_x),
                  'point_residuals': [p['fit_residual'] for p in points]}, indent=2))
