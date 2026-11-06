"""Staged genes recovered by a plain random forest, as a heatmap.

Same layout as the feature_comparison_wide_multi heatmap, with RF selection
rules as the columns: the RF top --n-top genes, and every gene with
importance above each of --thresholds. Uses the stability runs from
feature_comparison_rf_stability.py (datasets 0..9, RF seeds 42..46); the
fits are repeated here with the same seeds because that script keeps only
staged-gene importances.

Heatmap cells show the mean over datasets with the SEM across the per-dataset
means (each the mean of the RF seeds), as in the multi heatmap. Column labels
give the mean number of genes each rule returns.

Outputs (to --out-dir):
  rf_heatmap_counts.csv     one row per dataset, scenario, seed and rule
  rf_heatmap.png / .pdf

Usage:
  python feature_comparison_rf_heatmap.py
  python feature_comparison_rf_heatmap.py --plot-only
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_comparision import (BUILDERS, SCENARIO_ORDER, check_staging, _mixed_text,
								 _RAMP, _INK, _MUTED, _SURFACE)
from feature_comparison_rf_test import run_rf


def rule_names(n_top, thresholds):
	return [f"Top {n_top}"] + [f"> {t:g}" for t in thresholds]


def plot_rf_heatmap(df, out_dir, n_staged, rules):
	"""Mean staged genes recovered per scenario and RF rule, SEM across datasets."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap

	plt.rcParams.update({
		"font.family": "Arial",
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	scenarios = [s for s in SCENARIO_ORDER if s in set(df["scenario"])]
	n_ds, n_seeds = df["dataset"].nunique(), df["seed"].nunique()
	per_ds = df.groupby(["scenario", "rule", "dataset"], as_index=False)[
		["n_recovered", "panel_size"]].mean()
	M = per_ds.pivot_table(index="scenario", columns="rule", values="n_recovered")
	M = M.reindex(index=scenarios, columns=rules).to_numpy(dtype=float)
	S = per_ds.pivot_table(index="scenario", columns="rule", values="n_recovered", aggfunc="sem")
	S = np.nan_to_num(S.reindex(index=scenarios, columns=rules).to_numpy(dtype=float), nan=0.0)
	sizes = df.groupby("rule")["panel_size"].mean().reindex(rules)

	cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
	fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * M.shape[1] + 2.4),
									 0.85 * M.shape[0] + 1.9))
	fig.subplots_adjust(left=0.22, right=0.86, top=0.88, bottom=0.31)
	fig.patch.set_facecolor(_SURFACE); ax.set_facecolor(_SURFACE)
	mesh = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=n_staged,
						 edgecolors=_SURFACE, linewidth=2)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = M[i, j]
			label = "0.0" if abs(v) < 1e-12 else f"{v:.1f}"
			color = _SURFACE if v > n_staged * 0.55 else _INK
			ax.text(j + 0.5, i + 0.42, label, ha="center", va="center", fontsize=8, color=color)
			ax.text(j + 0.5, i + 0.68, f"±{S[i, j]:.2f}", ha="center",
					va="center", fontsize=6, color=color)

	# red box around the top-n column, the panel size the multi heatmap uses
	from matplotlib.patches import Rectangle
	ax.add_patch(Rectangle((0.03, 0.03), 0.94, M.shape[0] - 0.06, fill=False,
						   edgecolor="#c8102e", linewidth=2.5, clip_on=False, zorder=5))

	ax.set_xticks(np.arange(M.shape[1]) + 0.5)
	# small panels keep a decimal, so 0.9 genes does not read as 1
	ax.set_xticklabels([f"{r}\n({sizes[r]:.{1 if sizes[r] < 10 else 0}f} genes)" for r in rules],
					   rotation=35,
					   ha="right", fontsize=12, color=_INK)
	ax.set_yticks(np.arange(M.shape[0]) + 0.5)
	ax.set_yticklabels(scenarios, fontsize=13, color=_INK)
	ax.invert_yaxis()
	for sp in ax.spines.values():
		sp.set_visible(False)
	ax.tick_params(length=0)

	cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
	cbar.set_ticks(np.arange(0, n_staged + 1, 2))
	_mixed_text(fig, [("mean genes recovered ", {}), (f"(of {n_staged})", {"fontstyle": "italic"})],
				4.2, 0.5, cbar.ax.transAxes, vertical=True, fontsize=12, color=_INK)
	cbar.outline.set_visible(False)
	cbar.ax.tick_params(length=0, labelsize=15, colors=_INK)

	_mixed_text(fig, [("Genes Recovered by Random Forest ", {"fontweight": "bold"}),
					  (f"(Simulated, {n_ds} datasets)", {"fontstyle": "italic"})],
				0.5, 1.05, ax.transAxes, fontsize=14, color=_INK)
	note = (f"Values are mean ± SEM across {n_ds} dataset means ({n_seeds} RF seeds each). "
			"Columns are RF gene panels: the top genes by impurity importance, or every gene "
			"above an importance cutoff; the number under each is the mean panel size. "
			"Plain RF (300 trees) on 2000 cells × 2000 genes; an even split of importance "
			"is 0.0005 per gene.")
	fig.text(0.5, 0.02, textwrap.fill(note, 125), ha="center", va="bottom", fontsize=8,
			 color=_MUTED)
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"rf_heatmap.{ext}", dpi=300, facecolor=_SURFACE)
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--datasets", type=int, default=10)
	ap.add_argument("--seeds", type=int, default=5, help="RF seeds per dataset")
	ap.add_argument("--rf-seed", type=int, default=42, help="seed i uses rf_seed + i")
	ap.add_argument("--n-top", type=int, default=50)
	ap.add_argument("--thresholds", default="0.01,0.005,0.002,0.001")
	ap.add_argument("--n-staged", type=int, default=10)
	ap.add_argument("--ncells", type=int, default=2000)
	ap.add_argument("--ngenes", type=int, default=2000)
	ap.add_argument("--sim-seed", type=int, default=32)
	ap.add_argument("--stage-seed", type=int, default=0)
	ap.add_argument("--out-dir", default=str(
		Path(__file__).resolve().parent / "robustness" / "feature_comparison_rf_test"))
	ap.add_argument("--plot-only", action="store_true")
	args = ap.parse_args()

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	thresholds = [float(t) for t in args.thresholds.split(",")]
	rules = rule_names(args.n_top, thresholds)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	csv = out_dir / "rf_heatmap_counts.csv"

	if not args.plot_only:
		rows = []
		for ds in range(args.datasets):
			for scen in scenarios:
				adata, staged = BUILDERS[scen](
					sim_seed=args.sim_seed + ds, stage_seed=args.stage_seed + ds,
					n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes)
				check_staging(adata, staged)
				for seed in range(args.seeds):
					t0 = time.time()
					imp, _ = run_rf(adata, args.rf_seed + seed)
					panels = [imp.sort_values(ascending=False).index[:args.n_top]]
					panels += [imp.index[imp > t] for t in thresholds]
					line = []
					for rule, panel in zip(rules, panels):
						n = int(panel.isin(staged).sum())
						rows.append(dict(dataset=ds, scenario=scen, seed=seed, rule=rule,
										 n_recovered=n, panel_size=len(panel)))
						line.append(f"{rule}={n}/{len(panel)}")
					print(f"  [d{ds} {scen} s{seed}] " + "  ".join(line)
						  + f"   [{time.time()-t0:.0f}s]", flush=True)
			# save after each dataset so a crash loses little
			pd.DataFrame(rows).to_csv(csv, index=False)

	df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
	piv = df.groupby(["scenario", "rule"])["n_recovered"].mean().unstack()
	order = [s for s in SCENARIO_ORDER if s in piv.index]
	print("mean staged genes recovered")
	print(piv.reindex(index=order, columns=rules).round(2).to_string())
	plot_rf_heatmap(df, out_dir, args.n_staged, rules)


if __name__ == "__main__":
	main()
