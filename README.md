# Data and plotting scripts for "Unraveling square and kagome lattice frustrated magnetism by scalable tensor networks"

Xing-Yu Zhang, Jutho Haegeman, Yuchi He (Ghent University)

arXiv: [2611.09881](https://arxiv.org/abs/2611.09881)

This repository contains the data shown in the figures of the paper and its
Supplementary Information, together with the Python scripts that draw them. Only
values that appear in a figure, or enter a fit drawn in one, are included. The
iPEPS optimization and contraction code is the open-source Julia package
[TeneT.jl](https://github.com/XingyuZhang2018/TeneT.jl).

## Layout

```
figures/
  plot_*.py                   plotting scripts (see the table below)
  data/                       figure data (CSV / TSV)
  fig8_pq_scan_D8_chi128.tsv  data for Supp. Fig. 6
  published/                  the figure PDFs exactly as they appear in the paper
  J2_results_stacked.pdf      assembled Fig. 1 (input of plot_fig1_order_axes.py)
```

## Reproducing the figures

Python 3.10 or newer:

```sh
pip install -r requirements.txt
python figures/plot_square_kagome_log_m2_vs_log_inv_xi_kagome_chi_ge_3D2.py
```

Each script writes its figure next to itself in `figures/`; compare with the
reference copy in `figures/published/`. The scripts use the Arial font;
matplotlib substitutes a default font where Arial is not installed.

## Figure → script → data

### Main text

| Figure | Script | Data (`figures/data/`) |
|---|---|---|
| Fig. 1a,b — Néel / plaquette order vs. J2/J1; m² vs. 1/ξ extrapolation | `plot_J2_results_stacked.py` (panels a,b standalone); `plot_fig1_order_axes.py`, `plot_fig1c_legend.py` (redraw panel a / legend inside the assembled `J2_results_stacked.pdf`) | `J2_sweep_fit_points.csv`, `J2_sweep_fit_results.csv`, `J2_sweep_vbs_order.csv` |
| Fig. 1c — J2/J1 = 0.5 energy extrapolation and comparison | `plot_J05_E_extrap.py` (standalone panel; the paper uses a wider layout of the same data) | `J05_E_extrap_ipeps.csv`, `J05_E_extrap_references.csv` |
| Fig. 2 — ansatz and contraction | schematic, no data | — |
| Fig. 3 — kagome energy vs. 1/ξ³ for different unit cells | `plot_square_kagome_log_m2_vs_log_inv_xi_kagome_chi_ge_3D2.py` (also needs `plot_kagome_merge_compact.py`) | `Square_J1J2_J05_vs_HB_Kagome_log_m2_vs_log_inv_xi_kagome_chi_ge_3D2.csv`, `kagome_2x2_optimized_energy_xi.csv`, `kagome_energy_benchmarks_arxiv_2512_18831.csv`, `kagome_unit_cell_bonds.csv`, `kagome_unit_cell_magnetization.csv` |
| Fig. 4 — log m² vs. log 1/ξ, square J1-J2, square Heisenberg, kagome | same script as Fig. 3 | `Square_J1J2_J05_vs_HB_Kagome_log_m2_vs_log_inv_xi_kagome_chi_ge_3D2.csv`, `kagome_3pess_su_min_magnetization.csv` |
| Fig. 5 — square-lattice Heisenberg: iPEPS vs. QMC | `plot_heisenberg_qmc_ipeps_extrapolation.py` | `heisenberg_qmc_ipeps_extrapolation_ipeps.csv`, `heisenberg_qmc_ipeps_extrapolation_qmc_table.csv`, `heisenberg_qmc_ipeps_extrapolation_qmc_parameters.csv` |
| Fig. 6 — FLmap contraction | schematic, no data | — |

### Supplementary Information

| Figure | Script | Data (`figures/data/`) |
|---|---|---|
| Supp. Fig. 1 — plaquette order at J2/J1 = 0.55 vs. 1/D | `plot_J055_vbs_max_chi_vs_invD_extrap.py` | `Square_J055_Fig6_max_chi_points.csv`, `Square_J055_Fig6_invxi_points.csv` |
| Supp. Fig. 2 — bond isotropization, unrestricted π-mirror ansatz | `plot_square_j055_d10_unrestricted_bond_isotropization.py` | `Square_J055_D10_unrestricted_bond_isotropization.csv` |
| Supp. Fig. 3 — kagome 2×2 bond-energy range scaling | `plot_vbs_log_scaling.py` | `kagome_2x2_bond_abs_range_chi1024.csv` |
| Supp. Fig. 4 — re-evaluation at larger χ | `plot_heisenberg_chi_reoptimization.py` | `heisenberg_chi_reoptimization.csv` and the Fig. 5 files (it reuses the Fig. 5 fit) |
| Supp. Fig. 5 — randomized loop contraction | schematic, no data | — |
| Supp. Fig. 6 — projector accuracy and speed-up, D = 8, χ = 128 | `plot_figS6_projector_validation.py` | `../fig8_pq_scan_D8_chi128.tsv` |
| Supp. Fig. 7 — projector error, D = 8 and D = 16 | `plot_fig8_projector_total_energy_error_p_D8_D16.py` | `fig8_projector_total_energy_error_p_D8_D16.tsv` |
| Supp. Figs. 8, 9 — slicing and MPI collectives | schematics, no data | — |
| Supp. Fig. 10 — multi-GPU strong scaling | `plot_multigpu_scaling.py` | `multigpu_scaling_flmap.csv` (selected timings; speedup = `baseline_time_ms / time_ms`) |

## Third-party data

- `kagome_3pess_su_min_magnetization.csv`: the smallest magnetization of the
  projected entangled-simplex state (PESS) data for the kagome Heisenberg model,
  provided by Hai-Jun Liao.
- `kagome_energy_benchmarks_arxiv_2512_18831.csv`,
  `J05_E_extrap_references.csv`, `heisenberg_qmc_ipeps_extrapolation_qmc_*.csv`:
  values taken from the published works cited in the paper.

Please cite the original works when reusing these values.

## License

Scripts: MIT (see `LICENSE`). Data: CC BY 4.0 (see `LICENSE-DATA`), except the
third-party values listed above, which remain with their original authors.

## Citation

If you use these data, please cite the paper:

```bibtex
@misc{Zhang2026ScalableTN,
  author        = {Xing-Yu Zhang and Jutho Haegeman and Yuchi He},
  title         = {Unraveling square and kagome lattice frustrated magnetism by scalable tensor networks},
  year          = {2026},
  eprint        = {2611.09881},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  url           = {https://arxiv.org/abs/2611.09881}
}
```
