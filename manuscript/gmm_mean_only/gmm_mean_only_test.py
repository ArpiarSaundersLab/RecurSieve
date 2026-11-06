"""RecurSieve with the GMM fit on the panel mean only (no library size).

The default GMM clusters cells on two standardized columns: log1p_total_counts
and the mean expression of the selected panel. This test drops the
log1p_total_counts column, so the GMM is a 1-D, 2-component fit on the panel
mean. Everything else (RF, stopping rules, first iteration) is unchanged.

A third variant, "mean only, no stop", is the mean-only GMM with the
threshold, oscillation and patience stops switched off, so every run does
max_iterations iterations (as in coexpression_null). It separates two
explanations for a mean-only gap: worse GMM labels, or stopping too early.

Both variants run on the same simulated datasets and seeds as
feature_comparision.py (dataset d uses sim_seed + d and stage_seed + d; seed i
uses rs_seed + i), so the comparison is paired. Builders are imported from
feature_comparision.py unchanged.

Outputs (to --out-dir, default results/ next to this script):
  gmm_mean_only_counts.csv   one row per dataset, scenario, seed and variant
  gmm_mean_only_heatmap.png / .pdf

Usage:
  python gmm_mean_only_test.py
  python gmm_mean_only_test.py --datasets 10 --seeds 5
  python gmm_mean_only_test.py --plot-only
  python gmm_mean_only_test.py --datasets 2 --seeds 3 --variants "mean only, no stop"
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import contextlib
import io
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# builders and plot style come from the existing benchmark, unchanged
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_comparision import (BUILDERS, SCENARIO_ORDER, check_staging, count_recovered,
								 _mixed_text, _RAMP, _INK, _MUTED, _SURFACE)
import recursieve

VARIANTS = ["counts + mean", "mean only", "mean only, no stop"]


class recursieve_mean_only(recursieve.recursieve):
	"""recursieve whose GMM sees only the panel mean, not library size."""

	def cluster_infectivity_gmm(self, gene=None):
		gene = self.genes[-1]
		if self.additive:
			expr = self._summarize_selected().reshape(-1, 1)
			self.gene_expression_log["summed"] = expr.ravel()
			self.adata.obs["rf_summed"] = expr.ravel()
		else:
			expr = np.asarray(self.adata[:, gene].X).reshape(-1, 1)

		# same as the default, minus the log1p_total_counts column
		X = StandardScaler().fit_transform(expr)
		gmm = GaussianMixture(n_components=2, random_state=self.seed)
		labels = gmm.fit_predict(X)

		# label cluster with higher expr as "1"
		means = [expr[labels == k].mean() for k in [0, 1]]
		hi = int(np.argmax(means))
		labels = (labels == hi).astype(int)

		self.adata.obs[f"gmm_{gene}"] = labels.astype(str)
		self.gmm = gmm
		return gmm, labels


def run_variant(adata, variant, seed, n_top=50, max_iterations=50, n_jobs=-1):
	"""Top n_top genes of one recursieve run; returns (panel, stop_reason)."""
	cls = recursieve_mean_only if variant.startswith("mean only") else recursieve.recursieve
	# no stop: threshold, oscillation and patience stops all switched off
	stops = (dict(flip_rate_percentage=-1.0, patience=None) if variant.endswith("no stop")
			 else dict(flip_rate_percentage=0.01))
	# silence the per-iteration prints
	with contextlib.redirect_stdout(io.StringIO()):
		model = cls(adata.copy(), group1="group_1", group2="group_2", field_name="group",
					max_iterations=max_iterations, seed=seed, n_jobs=n_jobs,
					print_to_console=False, plots=False, **stops)
	return list(model.genes)[:n_top], model.stop_reason


def run_job(ds, scen, seed, args, n_jobs):
	"""Build one dataset and run the chosen variants on it with one seed."""
	adata, staged = BUILDERS[scen](
		sim_seed=args.sim_seed + ds, stage_seed=args.stage_seed + ds,
		n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes)
	check_staging(adata, staged)
	rows = []
	for v in args.variants:
		t0 = time.time()
		panel, stop = run_variant(adata, v, args.rs_seed + seed, n_top=args.n_top,
								  max_iterations=args.max_iterations, n_jobs=n_jobs)
		rows.append(dict(dataset=ds, scenario=scen, seed=seed, variant=v,
						 n_recovered=count_recovered(panel, staged), panel_size=len(panel),
						 stop_reason=stop, seconds=round(time.time() - t0, 1)))
	line = "  ".join(f"{r['variant']}={r['n_recovered']}/{r['panel_size']}" for r in rows)
	print(f"  [d{ds} {scen} s{seed}] {line}", flush=True)
	return rows


def plot_heatmap(df, out_dir, n_staged):
	"""Mean staged genes recovered per scenario and GMM variant, SEM across datasets."""
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
	per_ds = df.groupby(["scenario", "variant", "dataset"], as_index=False)["n_recovered"].mean()
	M = per_ds.pivot_table(index="scenario", columns="variant", values="n_recovered")
	variants = [v for v in VARIANTS if v in set(df["variant"])]
	M = M.reindex(index=scenarios, columns=variants).to_numpy(dtype=float)
	S = per_ds.pivot_table(index="scenario", columns="variant", values="n_recovered", aggfunc="sem")
	S = np.nan_to_num(S.reindex(index=scenarios, columns=variants).to_numpy(dtype=float), nan=0.0)

	cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
	fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * M.shape[1] + 2.4), 0.85 * M.shape[0] + 1.9))
	fig.subplots_adjust(left=0.22, right=0.86, top=0.88, bottom=0.36)
	fig.patch.set_facecolor(_SURFACE); ax.set_facecolor(_SURFACE)
	mesh = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=n_staged, edgecolors=_SURFACE, linewidth=2)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = M[i, j]
			label = "0.0" if abs(v) < 1e-12 else f"{v:.1f}"
			color = _SURFACE if v > n_staged * 0.55 else _INK
			ax.text(j + 0.5, i + 0.42, label, ha="center", va="center", fontsize=8, color=color)
			ax.text(j + 0.5, i + 0.68, f"±{S[i, j]:.2f}", ha="center",
					va="center", fontsize=6, color=color)

	ax.set_xticks(np.arange(M.shape[1]) + 0.5)
	ax.set_xticklabels([f"GMM: {v}" for v in variants], rotation=30, ha="right",
					   fontsize=12, color=_INK)
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

	_mixed_text(fig, [("RecurSieve GMM Input ", {"fontweight": "bold"}),
					  (f"(Simulated, {n_ds} datasets)", {"fontstyle": "italic"})],
				0.5, 1.05, ax.transAxes, fontsize=14, color=_INK)
	ds = sorted(df["dataset"].unique())
	note = (f"Values are mean ± SEM across {n_ds} dataset means ({n_seeds} RecurSieve seeds "
			f"each), top 50 genes; datasets {ds[0]}–{ds[-1]}, which recover more than the "
			"25-dataset average. counts + mean: default GMM on log1p total counts and the "
			"panel mean expression. mean only: 1-D GMM on the panel mean expression. no stop: "
			"threshold, oscillation and patience stops off, so every run does 50 iterations.")
	fig.text(0.5, 0.02, textwrap.fill(note, 110), ha="center", va="bottom", fontsize=8,
			 color=_MUTED)
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"gmm_mean_only_heatmap.{ext}", dpi=300, facecolor=_SURFACE)
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--datasets", type=int, default=10)
	ap.add_argument("--seeds", type=int, default=5)
	ap.add_argument("--rs-seed", type=int, default=42, help="seed i uses rs_seed + i")
	ap.add_argument("--n-top", type=int, default=50)
	ap.add_argument("--max-iterations", type=int, default=50)
	ap.add_argument("--n-staged", type=int, default=10)
	ap.add_argument("--ncells", type=int, default=2000)
	ap.add_argument("--ngenes", type=int, default=2000)
	ap.add_argument("--sim-seed", type=int, default=32)
	ap.add_argument("--stage-seed", type=int, default=0)
	ap.add_argument("--n-parallel", type=int, default=None,
					help="runs at once; cores are split between them (default cores // 3)")
	ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
	ap.add_argument("--variants", default="counts + mean;mean only",
					help=f"';'-separated, from {VARIANTS}")
	ap.add_argument("--plot-only", action="store_true")
	args = ap.parse_args()
	args.variants = [v.strip() for v in args.variants.split(";")]
	bad = [v for v in args.variants if v not in VARIANTS]
	if bad:
		raise SystemExit(f"unknown variants {bad}; choose from {VARIANTS}")

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	csv = out_dir / "gmm_mean_only_counts.csv"

	if not args.plot_only:
		cores = os.cpu_count() or 1
		n_parallel = args.n_parallel or max(1, cores // 3)
		n_jobs = max(1, cores // n_parallel)
		# new rows replace saved rows with the same keys; others are kept
		keys = ["dataset", "scenario", "seed", "variant"]
		saved = (pd.read_csv(csv, keep_default_na=False, na_values=[""]) if csv.exists()
				 else pd.DataFrame(columns=keys))
		rows = []
		for ds in range(args.datasets):
			jobs = [(ds, s, k) for s in scenarios for k in range(args.seeds)]
			out = Parallel(n_jobs=n_parallel)(
				delayed(run_job)(d, s, k, args, n_jobs) for d, s, k in jobs)
			rows += [r for rs in out for r in rs]
			# save after each dataset so a crash loses little
			out = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
			out.drop_duplicates(subset=keys, keep="last").to_csv(csv, index=False)

	df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
	piv = df.groupby(["scenario", "variant"])["n_recovered"].mean().unstack()
	order = [s for s in SCENARIO_ORDER if s in piv.index]
	print("mean staged genes recovered")
	print(piv.reindex(index=order, columns=[v for v in VARIANTS if v in piv.columns])
		  .round(2).to_string())
	plot_heatmap(df, out_dir, args.n_staged)


if __name__ == "__main__":
	main()
