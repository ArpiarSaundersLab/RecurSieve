"""Compare recursieve GMM labels to the Disease.Group labels across iterations.

The GMM only gives clusters 0/1 (1 = higher summary expression), so there is
no fixed link to "low"/"high". Cluster 1 means the same thing at every
iteration, so we score agreement with one fixed matching (GMM 1 = high) and
also report the better of the two matchings.

Usage:
	python alzheimers_labels.py [--max-iterations N] [--seed N] [--out-dir DIR]
		[--cell-type TYPE] [--n-cells N] [--balance-groups]

With --max-iterations 1 (default) it prints the first iteration only. With
more iterations it also writes (to DIR):
- alzheimers_labels_iterations.csv: agreement, ARI, NMI, flip rate per iteration
- alzheimers_labels_cells.csv: per-cell disagreement, label changes, first/last GMM label
- alzheimers_labels_celltypes.csv: mean disagreement by cell type and group
- alzheimers_labels.pdf/.png: agreement over iterations and per-cell histogram
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import recursieve
from alz_loading import FIELD_NAME, add_loading_args, load_alzheimers

GROUP_COLORS = {"low": "#2a78d6", "high": "#eb6834"}


def print_first_iteration(truth, gmm, obs, cell_type_key) -> None:
	"""Crosstab and agreement for one set of GMM labels."""
	print("\nCounts (rows = Disease.Group, cols = GMM cluster):")
	print(pd.crosstab(truth, gmm, rownames=["group"], colnames=["gmm"], margins=True))
	print("\nFraction of each group in each GMM cluster:")
	print(pd.crosstab(truth, gmm, rownames=["group"], colnames=["gmm"],
					  normalize="index").round(3))

	agree_hi = ((gmm == "1") == (truth == "high")).mean()
	print(f"\nAgreement if GMM 1 = high: {agree_hi:.3f}")
	print(f"Agreement if GMM 1 = low:  {1 - agree_hi:.3f}")
	print(f"Best agreement:            {max(agree_hi, 1 - agree_hi):.3f}")
	print(f"Majority-class baseline:   {pd.Series(truth).value_counts(normalize=True).max():.3f}")
	print(f"ARI: {adjusted_rand_score(truth, gmm):.4f}")
	print(f"NMI: {normalized_mutual_info_score(truth, gmm):.4f}")

	# the GMM also sees log1p_total_counts, so check if it tracks cell type
	if cell_type_key in obs:
		print(f"\nFraction of each {cell_type_key} in GMM cluster 1:")
		frac = pd.crosstab(obs[cell_type_key], gmm, normalize="index")
		print(frac.get("1", pd.Series(dtype=float)).round(3).sort_values())


def iteration_table(model, truth, labels, genes) -> pd.DataFrame:
	"""One row per iteration with agreement against the original labels."""
	is_high = truth == "high"
	rows = []
	for i, (gene, lab) in enumerate(zip(genes, labels)):
		agree = ((lab == 1) == is_high).mean()
		rows.append({
			"iteration": i + 1,
			"gmm_gene": gene,  # newest gene in the summary used by this GMM
			"n_genes_in_summary": i + 1,
			"agreement_1_is_high": agree,
			"agreement_best": max(agree, 1 - agree),
			"ari": adjusted_rand_score(is_high, lab),
			"nmi": normalized_mutual_info_score(is_high, lab),
			"frac_high_in_1": lab[is_high].mean(),
			"frac_low_in_1": lab[~is_high].mean(),
			# flip rate vs the previous iteration (none for the first)
			"flip_rate_vs_prev": model.flip_rates[i - 1] if 0 < i <= len(model.flip_rates) else np.nan,
		})
	return pd.DataFrame(rows)


def plot_summary(iters, cells, baseline, out_dir) -> None:
	"""Agreement over iterations, and the per-cell disagreement histogram."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

	ax1.plot(iters["iteration"], iters["agreement_1_is_high"], color="#2a78d6",
			 lw=2, marker="o", ms=4, label="GMM 1 = high")
	ax1.plot(iters["iteration"], iters["agreement_best"], color="#eb6834",
			 lw=2, marker="o", ms=4, label="Best matching")
	ax1.axhline(baseline, color="#8a8984", lw=1, ls="--", label="Majority class")
	ax1.axhline(0.5, color="#c9c8c2", lw=1)
	ax1.set_xlabel("Iteration")
	ax1.set_ylabel("Agreement with Disease.Group")
	ax1.set_ylim(0.3, 0.8)
	ax1.legend(frameon=False, fontsize=8)

	bins = np.linspace(0, 1, 21)
	for group, color in GROUP_COLORS.items():
		vals = cells.loc[cells["group"] == group, "frac_disagree"]
		ax2.hist(vals, bins=bins, color=color, alpha=0.6, label=group,
				 edgecolor="white", linewidth=1)
	ax2.set_xlabel("Fraction of iterations disagreeing (GMM 1 = high)")
	ax2.set_ylabel("Cells")
	ax2.legend(title="Disease.Group", frameon=False, fontsize=8)

	for ax in (ax1, ax2):
		ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_dir / "alzheimers_labels.pdf")
	fig.savefig(out_dir / "alzheimers_labels.png", dpi=200)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--max-iterations", type=int, default=1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	add_loading_args(parser)
	args = parser.parse_args()

	adata_path = Path(__file__).resolve().parent.parent / "data" / "alz.h5ad"
	adata = load_alzheimers(adata_path, args.cell_type, args.cell_type_key,
							args.n_cells, args.balance_groups)
	print(f"Data shape: {adata.n_obs} cells x {adata.n_vars} genes")

	model = recursieve.recursieve(
		adata=adata, group1="low", group2="high", field_name=FIELD_NAME,
		plots=False, print_to_console=False, max_iterations=args.max_iterations,
		seed=args.seed,
	)
	obs = model.adata.obs
	truth = obs[FIELD_NAME].astype(str).to_numpy()

	# each iteration stores its GMM labels in obs["gmm_<newest gene>"]
	genes = [g for g in model.genes if f"gmm_{g}" in obs]
	labels = [obs[f"gmm_{g}"].astype(int).to_numpy() for g in genes]
	print(f"\n{len(genes)} GMM iterations, {len(model.genes)} genes, "
		  f"stop reason: {model.stop_reason}")

	print(f"\nIteration 1 (GMM on {genes[0]})")
	print_first_iteration(truth, labels[0].astype(str), obs, args.cell_type_key)
	if len(genes) < 2:
		return

	out_dir = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	iters = iteration_table(model, truth, labels, genes)
	iters["stop_reason"] = model.stop_reason
	iters.to_csv(out_dir / "alzheimers_labels_iterations.csv", index=False)
	print("\nPer iteration:")
	print(iters.drop(columns="stop_reason").round(3).to_string(index=False))

	# per cell: how often its GMM label disagrees with its original label
	mat = np.vstack(labels)  # iterations x cells
	disagree = (mat == 1) != (truth == "high")
	cells = pd.DataFrame({
		"cell": obs.index,
		"group": truth,
		"frac_disagree": disagree.mean(axis=0),
		"n_label_changes": (np.diff(mat, axis=0) != 0).sum(axis=0),
		"gmm_first": mat[0],
		"gmm_last": mat[-1],
	})
	if args.cell_type_key in obs:
		cells["cell_type"] = obs[args.cell_type_key].astype(str).to_numpy()
	cells.to_csv(out_dir / "alzheimers_labels_cells.csv", index=False)

	print("\nPer-cell disagreement (fraction of iterations, GMM 1 = high):")
	bands = pd.cut(cells["frac_disagree"], [-0.01, 0.1, 0.9, 1.0],
				   labels=["mostly agree (<=0.1)", "switching", "mostly disagree (>0.9)"])
	print(pd.crosstab(cells["group"], bands, normalize="index").round(3))
	print("\nMean label changes per cell:", round(cells["n_label_changes"].mean(), 2))

	if "cell_type" in cells:
		by_type = cells.groupby(["cell_type", "group"])["frac_disagree"].mean().unstack()
		by_type.to_csv(out_dir / "alzheimers_labels_celltypes.csv")
		print("\nMean disagreement by cell type:")
		print(by_type.round(3))

	baseline = pd.Series(truth).value_counts(normalize=True).max()
	plot_summary(iters, cells, baseline, out_dir)
	print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
	main()
