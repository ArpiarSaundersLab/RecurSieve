"""Seed stability of the plain random forest on the feature comparison datasets.

Runs the RF from feature_comparison_rf_test.py with several seeds on each of
several datasets (2000 cells x 2000 genes by default). Dataset d uses
sim_seed + d and stage_seed + d, as in feature_comparision.py. The RF seed
also sets the 70/30 split, so seed i uses rf_seed + i for both.

Per run it records how many staged genes have importance above --threshold,
the largest staged and non-staged importances, and how many staged genes are
in the RF top --n-top. Stability is the mean pairwise Jaccard overlap of the
RF top --n-top genes across seeds on the same dataset.

Outputs (to --out-dir):
  rf_stability_runs.csv                one row per dataset, scenario and seed
  rf_stability_staged_importances.csv  importance and rank of every staged gene
  rf_stability_jaccard.csv             top-n overlap across seeds per dataset
  rf_stability_summary.csv             per-scenario summary
  rf_stability.png / .pdf

Usage:
  python feature_comparison_rf_stability.py
  python feature_comparison_rf_stability.py --datasets 3 --seeds 2
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_comparision import BUILDERS, SCENARIO_ORDER, check_staging, _RAMP, _INK, _MUTED, _SURFACE
from feature_comparison_rf_test import run_rf


def jaccard(a, b):
	a, b = set(a), set(b)
	return len(a & b) / len(a | b)


def plot_stability(runs, jac, out_dir, threshold, n_top, ngenes):
	"""Three panels: top staged importance, staged genes in top n, seed overlap."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
	scen_all = [s for s in SCENARIO_ORDER if s in set(runs["scenario"])]
	scen_staged = [s for s in scen_all if s != "null"]
	n_ds, n_seeds = runs["dataset"].nunique(), runs["seed"].nunique()
	dot, fill, edge = _RAMP[6], _RAMP[3], _RAMP[7]

	fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
	fig.patch.set_facecolor(_SURFACE)

	def dots(ax, groups, values_of, log=False):
		for j, s in enumerate(groups):
			v = values_of(s)
			jitter = np.random.default_rng(j).uniform(-0.22, 0.22, len(v))
			ax.scatter(j + jitter, v, s=12, color=dot, alpha=0.55, linewidths=0, zorder=3)
			ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=_INK, lw=2, zorder=4)
		ax.set_xticks(range(len(groups)))
		ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=10, color=_INK)
		ax.set_xlim(-0.6, len(groups) - 0.4)
		if log:
			ax.set_yscale("log")

	ax = axes[0]
	dots(ax, scen_staged, lambda s: runs.loc[runs.scenario == s, "max_staged_importance"], log=True)
	ax.axhline(threshold, color=_INK, lw=1.2, ls="--", zorder=2)
	ax.text(len(scen_staged) - 0.45, threshold * 1.12, f"threshold {threshold:g}",
			ha="right", va="bottom", fontsize=9, color=_INK)
	ax.axhline(1 / ngenes, color=_MUTED, lw=1, ls=":", zorder=2)
	ax.text(len(scen_staged) - 0.45, (1 / ngenes) * 0.88, f"even split 1/{ngenes}",
			ha="right", va="top", fontsize=9, color=_MUTED)
	ax.set_ylim(1 / ngenes * 0.5, threshold * 3)
	ax.set_title("Highest staged-gene importance", fontsize=11, color=_INK)
	ax.set_ylabel("RF importance (log)", fontsize=10, color=_INK)

	ax = axes[1]
	dots(ax, scen_staged, lambda s: runs.loc[runs.scenario == s, "n_staged_in_top"])
	ax.set_ylim(-0.4, 10.4)
	ax.set_yticks(range(0, 11, 2))
	ax.set_title(f"Staged genes in RF top {n_top}", fontsize=11, color=_INK)
	ax.set_ylabel("genes (of 10)", fontsize=10, color=_INK)

	ax = axes[2]
	dots(ax, scen_all, lambda s: jac.loc[jac.scenario == s, "mean_jaccard"])
	ax.set_ylim(0, 1.02)
	ax.set_title(f"Top {n_top} overlap across seeds", fontsize=11, color=_INK)
	ax.set_ylabel("mean pairwise Jaccard", fontsize=10, color=_INK)

	for ax in axes:
		ax.set_facecolor(_SURFACE)
		ax.grid(axis="y", color="#e4e3df", lw=0.8)
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color(_MUTED)
		ax.tick_params(colors=_MUTED, length=0)

	fig.suptitle("Plain Random Forest, Seed Stability", fontsize=13, fontweight="bold",
				 color=_INK, y=0.98)
	fig.text(0.5, 0.905, f"{n_ds} datasets × {n_seeds} RF seeds, 2000 cells × {ngenes} genes; "
			 f"left and middle: one dot per run; right: one dot per dataset; black bar: median",
			 ha="center", va="top", fontsize=9, color=_MUTED)
	n_above = int(runs["n_staged_above"].sum())
	fig.text(0.5, 0.02, f"Staged genes with importance > {threshold:g}: {n_above} across all "
			 f"{len(runs[runs.scenario != 'null'])} non-null runs. The null scenario has no staged genes.",
			 ha="center", va="bottom", fontsize=9, color=_MUTED)
	fig.tight_layout(rect=(0, 0.05, 1, 0.88))
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"rf_stability.{ext}", dpi=300, facecolor=_SURFACE)
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--datasets", type=int, default=10)
	ap.add_argument("--seeds", type=int, default=5, help="RF seeds per dataset")
	ap.add_argument("--rf-seed", type=int, default=42, help="seed i uses rf_seed + i")
	ap.add_argument("--threshold", type=float, default=0.01)
	ap.add_argument("--n-top", type=int, default=50)
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
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	runs_csv = out_dir / "rf_stability_runs.csv"
	jac_csv = out_dir / "rf_stability_jaccard.csv"

	if not args.plot_only:
		rows, imp_rows, jac_rows = [], [], []
		for ds in range(args.datasets):
			for scen in scenarios:
				adata, staged = BUILDERS[scen](
					sim_seed=args.sim_seed + ds, stage_seed=args.stage_seed + ds,
					n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes)
				check_staging(adata, staged)
				panels = []
				for seed in range(args.seeds):
					t0 = time.time()
					imp, auc = run_rf(adata, args.rf_seed + seed)
					rank = imp.rank(ascending=False, method="min").astype(int)
					top = imp.sort_values(ascending=False).index[:args.n_top]
					panels.append(top)
					above = imp[imp > args.threshold]
					n_above = int(above.index.isin(staged).sum())
					rows.append(dict(
						dataset=ds, scenario=scen, seed=seed, rf_seed=args.rf_seed + seed,
						test_auc=auc, n_staged_above=n_above,
						n_other_above=len(above) - n_above,
						n_staged_in_top=int(top.isin(staged).sum()),
						max_staged_importance=float(imp[staged].max()) if staged else np.nan,
						best_staged_rank=int(rank[staged].min()) if staged else np.nan,
						max_other_importance=float(imp.drop(staged).max()),
					))
					for g in staged:
						imp_rows.append(dict(dataset=ds, scenario=scen, seed=seed, gene=g,
											 importance=float(imp[g]), rank=int(rank[g])))
					r = rows[-1]
					print(f"  [d{ds} {scen} s{seed}] staged>{args.threshold:g}: {n_above}  "
						  f"other>{args.threshold:g}: {r['n_other_above']}  "
						  f"staged in top {args.n_top}: {r['n_staged_in_top']}  "
						  f"AUC {auc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
				pairs = [jaccard(a, b) for a, b in itertools.combinations(panels, 2)]
				jac_rows.append(dict(dataset=ds, scenario=scen, mean_jaccard=np.mean(pairs),
									 min_jaccard=np.min(pairs)))
			# save after each dataset so a crash loses little
			pd.DataFrame(rows).to_csv(runs_csv, index=False)
			pd.DataFrame(imp_rows).to_csv(out_dir / "rf_stability_staged_importances.csv", index=False)
			pd.DataFrame(jac_rows).to_csv(jac_csv, index=False)

	runs = pd.read_csv(runs_csv, keep_default_na=False, na_values=[""])
	jac = pd.read_csv(jac_csv, keep_default_na=False, na_values=[""])
	order = [s for s in SCENARIO_ORDER if s in set(runs["scenario"])]
	# seed spread is the SD across seeds within a dataset, averaged over datasets
	seed_sd = runs.groupby(["scenario", "dataset"])[["n_staged_in_top", "max_staged_importance"]].std()
	summary = runs.groupby("scenario").agg(
		runs=("seed", "size"),
		staged_above_total=("n_staged_above", "sum"),
		runs_with_any_staged_above=("n_staged_above", lambda v: int((v > 0).sum())),
		mean_other_above=("n_other_above", "mean"),
		mean_staged_in_top=("n_staged_in_top", "mean"),
		median_max_staged_importance=("max_staged_importance", "median"),
		max_max_staged_importance=("max_staged_importance", "max"),
		median_best_staged_rank=("best_staged_rank", "median"),
		mean_test_auc=("test_auc", "mean"),
	)
	summary["seed_sd_staged_in_top"] = seed_sd.groupby("scenario")["n_staged_in_top"].mean()
	summary["seed_sd_max_staged_importance"] = seed_sd.groupby("scenario")["max_staged_importance"].mean()
	summary["mean_top_jaccard"] = jac.groupby("scenario")["mean_jaccard"].mean()
	summary = summary.reindex(order)
	summary.to_csv(out_dir / "rf_stability_summary.csv")
	print(f"\nSummary ({runs['dataset'].nunique()} datasets x {runs['seed'].nunique()} seeds)")
	print(summary.round(4).T.to_string())
	plot_stability(runs, jac, out_dir, args.threshold, args.n_top, args.ngenes)


if __name__ == "__main__":
	main()
