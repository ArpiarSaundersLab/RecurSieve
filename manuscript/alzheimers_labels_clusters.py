"""Place recursieve GMM labels and Disease.Group in an unsupervised clustering.

Uses recursieve's own preprocessing (same cells and 2000 HVGs the GMM saw),
then PCA, a kNN graph, Leiden clusters, and UMAP. For each Leiden cluster we
compare the fraction of high-pathology cells with the fraction of cells in
GMM cluster 1, and check where the label-switching cells sit.

Run alzheimers_labels.py first with the same cell options; this script reads
its alzheimers_labels_cells.csv from --out-dir.

Usage:
	python alzheimers_labels_clusters.py [--out-dir DIR] [--resolution R]
		[--cell-type TYPE] [--n-cells N] [--balance-groups]

Outputs (to DIR):
- alzheimers_labels_clusters.csv: per Leiden cluster composition
- alzheimers_labels_clusters_cells.csv: per cell Leiden cluster and UMAP
- alzheimers_labels_clusters.pdf/.png: UMAPs and per-cluster comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import recursieve
from alz_loading import FIELD_NAME, add_loading_args, load_alzheimers

LOW, HIGH = "#2a78d6", "#eb6834"
STABLE, SWITCH = "#d9d8d2", "#6250d6"


def embed(adata, resolution: float, seed: int) -> None:
	"""PCA, kNN graph, Leiden, and UMAP, stored on adata."""
	sc.pp.pca(adata, n_comps=50, random_state=seed)
	sc.pp.neighbors(adata, n_neighbors=15, random_state=seed)
	sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2,
				 random_state=seed)
	sc.tl.umap(adata, random_state=seed)


def cluster_table(cells: pd.DataFrame) -> pd.DataFrame:
	"""Composition of each Leiden cluster."""
	g = cells.groupby("leiden", observed=True)
	out = pd.DataFrame({
		"n_cells": g.size(),
		"frac_high": g["group"].apply(lambda s: (s == "high").mean()),
		"frac_gmm1_last": g["gmm_last"].mean(),
		"frac_gmm1_first": g["gmm_first"].mean(),
		"frac_switching": g["switching"].mean(),
		"mean_frac_disagree": g["frac_disagree"].mean(),
		"n_donors": g["donor"].nunique(),
		# share of the cluster from its largest donor (high = donor-specific)
		"top_donor_share": g["donor"].apply(lambda s: s.value_counts(normalize=True).iloc[0]),
	})
	if "subtype" in cells:
		out["top_subtype"] = g["subtype"].agg(lambda s: s.value_counts().index[0])
	return out.reset_index()


def scatter_umap(ax, xy, values, colors, labels, title) -> None:
	"""UMAP colored by a categorical column, drawn in shuffled order."""
	order = np.random.default_rng(0).permutation(len(xy))
	cmap = np.array([colors[v] for v in values])
	ax.scatter(xy[order, 0], xy[order, 1], c=cmap[order], s=1, linewidths=0, rasterized=True)
	for v, lab in labels.items():
		ax.scatter([], [], c=colors[v], s=20, label=lab)
	ax.legend(frameon=False, fontsize=7, markerscale=1, loc="best")
	ax.set_title(title, fontsize=10)


def plot_summary(cells, xy, clusters, out_dir) -> None:
	"""UMAPs by Leiden, disease, GMM and switching, plus per-cluster scatters."""
	fig, axes = plt.subplots(2, 3, figsize=(14, 9))
	ax = axes.ravel()

	# Leiden: neutral points with cluster numbers at the centroids
	ax[0].scatter(xy[:, 0], xy[:, 1], c="#b5b4ad", s=1, linewidths=0, rasterized=True)
	for c in clusters["leiden"]:
		cx, cy = np.median(xy[cells["leiden"] == c], axis=0)
		ax[0].text(cx, cy, str(c), fontsize=8, ha="center", va="center", weight="bold")
	ax[0].set_title("Leiden clusters", fontsize=10)

	scatter_umap(ax[1], xy, cells["group"], {"low": LOW, "high": HIGH},
				 {"low": "low", "high": "high"}, "Disease.Group")
	scatter_umap(ax[2], xy, cells["gmm_last"], {0: LOW, 1: HIGH},
				 {0: "GMM 0", 1: "GMM 1"}, "GMM label (final iteration)")
	scatter_umap(ax[3], xy, cells["switching"], {False: STABLE, True: SWITCH},
				 {False: "stable", True: "switching"}, "Label switching across iterations")

	# per cluster: does GMM composition follow disease composition?
	size = 10 + 200 * clusters["n_cells"] / clusters["n_cells"].max()
	ax[4].scatter(clusters["frac_high"], clusters["frac_gmm1_last"], s=size,
				  color=HIGH, alpha=0.7, edgecolor="white", linewidth=1)
	ax[4].set_xlabel("Fraction high-pathology cells")
	ax[4].set_ylabel("Fraction in GMM cluster 1 (final)")
	ax[4].set_title("Per Leiden cluster", fontsize=10)

	ax[5].scatter(clusters["frac_high"], clusters["frac_switching"], s=size,
				  color=SWITCH, alpha=0.7, edgecolor="white", linewidth=1)
	ax[5].set_xlabel("Fraction high-pathology cells")
	ax[5].set_ylabel("Fraction switching cells")
	ax[5].set_title("Per Leiden cluster", fontsize=10)

	for _, row in clusters.iterrows():
		for k, y in ((4, "frac_gmm1_last"), (5, "frac_switching")):
			ax[k].annotate(str(row["leiden"]), (row["frac_high"], row[y]),
						   fontsize=7, ha="center", va="center")

	for a in ax[:4]:
		a.set_xticks([]); a.set_yticks([])
		a.set_xlabel("UMAP1", fontsize=8); a.set_ylabel("UMAP2", fontsize=8)
	for a in ax:
		a.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_dir / "alzheimers_labels_clusters.pdf", dpi=200)
	fig.savefig(out_dir / "alzheimers_labels_clusters.png", dpi=200)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	parser.add_argument("--resolution", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=42)
	add_loading_args(parser)
	args = parser.parse_args()

	labels_path = args.out_dir / "alzheimers_labels_cells.csv"
	labels = pd.read_csv(labels_path, index_col="cell")
	labels.index = labels.index.astype(str)

	adata_path = Path(__file__).resolve().parent.parent / "data" / "alz.h5ad"
	adata = load_alzheimers(adata_path, args.cell_type, args.cell_type_key,
							args.n_cells, args.balance_groups)

	# same preprocessing and group filter as the recursieve run
	model = recursieve.recursieve(adata=adata, group1="low", group2="high",
								  field_name=FIELD_NAME, run=False)
	model.preprocessing()
	model.filter_groups()
	adata = model.adata
	if not adata.obs_names.isin(labels.index).all() or len(labels) != adata.n_obs:
		raise ValueError(f"cells do not match {labels_path}; use the same cell options")
	print(f"Clustering {adata.n_obs} cells x {adata.n_vars} genes")

	embed(adata, args.resolution, args.seed)

	obs = adata.obs
	cells = labels.loc[adata.obs_names.astype(str)].copy()
	cells["leiden"] = obs["leiden"].astype(int).to_numpy()
	cells["donor"] = obs["donor_id"].astype(str).to_numpy()
	if "Author_Annotation" in obs:
		cells["subtype"] = obs["Author_Annotation"].astype(str).to_numpy()
	cells["switching"] = (cells["frac_disagree"] > 0.1) & (cells["frac_disagree"] <= 0.9)
	xy = adata.obsm["X_umap"]
	cells["umap1"], cells["umap2"] = xy[:, 0], xy[:, 1]

	out_dir = args.out_dir
	cells.to_csv(out_dir / "alzheimers_labels_clusters_cells.csv", index_label="cell")
	clusters = cluster_table(cells)
	clusters.to_csv(out_dir / "alzheimers_labels_clusters.csv", index=False)

	print(f"\n{len(clusters)} Leiden clusters (resolution {args.resolution})")
	print(clusters.round(3).to_string(index=False))

	# how much does the unsupervised clustering line up with each labeling?
	print("\nAgreement with Leiden clusters:")
	for name, col in (("Disease.Group", "group"), ("GMM first", "gmm_first"),
					  ("GMM final", "gmm_last"), ("Switching", "switching"),
					  ("Donor", "donor")):
		print(f"  {name:<14} ARI={adjusted_rand_score(cells['leiden'], cells[col]):.4f}  "
			  f"NMI={normalized_mutual_info_score(cells['leiden'], cells[col]):.4f}")

	rho, p = spearmanr(clusters["frac_high"], clusters["frac_gmm1_last"])
	print(f"\nAcross clusters, frac high vs frac GMM 1 (final): Spearman rho={rho:.3f}, p={p:.3g}")
	rho, p = spearmanr(clusters["frac_high"], clusters["frac_switching"])
	print(f"Across clusters, frac high vs frac switching:     Spearman rho={rho:.3f}, p={p:.3g}")

	plot_summary(cells, xy, clusters, out_dir)
	print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
	main()
