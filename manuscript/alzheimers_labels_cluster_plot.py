"""Per-cluster summary of disease, GMM labels, switching, and donors.

Reads the outputs of alzheimers_labels_clusters.py from --out-dir and draws
one row per Leiden cluster, sorted by the fraction of high-pathology cells:
- left: % high cells vs % in GMM cluster 1 (final iteration), and % of cells
  whose own GMM label matches their Disease.Group label
- middle: % of cells that switch GMM labels across iterations
- right: share of each donor's cells that fall in the cluster

Usage:
	python alzheimers_labels_cluster_plot.py [--out-dir DIR]

Outputs (to DIR):
- alzheimers_labels_cluster_plot.pdf/.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOW, HIGH = "#2a78d6", "#eb6834"
GMM, SWITCH, GRID = "#2b2a27", "#6250d6", "#c9c8c2"
MATCH = "#1baf7a"


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	args = parser.parse_args()

	clusters = pd.read_csv(args.out_dir / "alzheimers_labels_clusters.csv")
	cells = pd.read_csv(args.out_dir / "alzheimers_labels_clusters_cells.csv")

	# per cell: does its final GMM label match its own disease label (GMM 1 = high)?
	cells["match"] = (cells["gmm_last"] == 1) == (cells["group"] == "high")
	clusters["frac_match"] = clusters["leiden"].map(cells.groupby("leiden")["match"].mean())

	# rows sorted so the most high-enriched cluster is on top
	clusters = clusters.sort_values("frac_high").reset_index(drop=True)
	y = np.arange(len(clusters))
	names = []
	for _, row in clusters.iterrows():
		sub = row["top_subtype"].split(":")[0] if "top_subtype" in clusters else ""
		names.append(f"Cluster {row['leiden']}  {sub}  (n={row['n_cells']:,})")

	# share of each donor's cells in each cluster
	donor_share = pd.crosstab(cells["donor"], cells["leiden"], normalize="index")
	donor_group = cells.groupby("donor")["group"].first()

	fig, ax = plt.subplots(1, 3, figsize=(14, 0.42 * len(clusters) + 3), sharey=True,
						   gridspec_kw={"width_ratios": [1.3, 0.9, 1.3]})
	below = dict(frameon=False, fontsize=8, loc="upper center",
				 bbox_to_anchor=(0.5, -0.09), ncol=1)

	# A: disease vs GMM composition
	pct_high = 100 * clusters["frac_high"]
	pct_gmm = 100 * clusters["frac_gmm1_last"]
	ax[0].hlines(y, pct_high, pct_gmm, color=GRID, lw=2)
	ax[0].scatter(pct_high, y, color=HIGH, s=60, zorder=3,
				  label="High-pathology cells (Disease.Group label)")
	ax[0].scatter(pct_gmm, y, color=GMM, marker="D", s=40, zorder=3,
				  label="Cells in GMM cluster 1 (RecurSieve, final iteration)")
	ax[0].scatter(100 * clusters["frac_match"], y, color=MATCH, marker="s", s=40, zorder=4,
				  edgecolor="white", linewidth=1,
				  label="Cells whose GMM label matches their own disease label (50% = chance)")
	ax[0].axvline(50, color=GRID, lw=1, ls="--", zorder=0)
	ax[0].set_xlim(20, 100)
	ax[0].set_xlabel("Percent of cells in the cluster")
	ax[0].set_yticks(y, names, fontsize=8)
	ax[0].set_title("A. Does the GMM label follow disease?", loc="left", fontsize=10)
	ax[0].legend(**below)

	# B: switching cells, with values inside the bar ends
	pct_switch = 100 * clusters["frac_switching"]
	ax[1].barh(y, pct_switch, color=SWITCH, height=0.6)
	for yi, v in zip(y, pct_switch):
		ax[1].text(v - 0.4, yi, f"{v:.0f}%", fontsize=7, va="center", ha="right", color="white")
	overall = 100 * cells["switching"].mean()
	ax[1].axvline(overall, color=GMM, lw=1, ls="--", label=f"All cells ({overall:.0f}%)")
	ax[1].set_xlim(0, pct_switch.max() * 1.2)
	ax[1].set_xlabel("Percent of cells in the cluster")
	ax[1].set_title("B. Cells whose GMM label switches", loc="left", fontsize=10)
	ax[1].legend(**below)

	# C: one dot per donor, jittered within the row
	rng = np.random.default_rng(0)
	for i, c in enumerate(clusters["leiden"]):
		share = 100 * donor_share[c]
		for group, color in (("low", LOW), ("high", HIGH)):
			vals = share[donor_group[share.index] == group]
			offset = -0.15 if group == "low" else 0.15
			ax[2].scatter(vals, i + offset + rng.uniform(-0.08, 0.08, len(vals)),
						  color=color, s=10, alpha=0.8, linewidths=0)
	ax[2].scatter([], [], color=LOW, s=20, label="Low-pathology donor")
	ax[2].scatter([], [], color=HIGH, s=20, label="High-pathology donor")
	ax[2].set_xlabel("Percent of the donor's cells in this cluster")
	ax[2].set_title("C. How much each donor contributes", loc="left", fontsize=10)
	ax[2].legend(**below)

	for a in ax:
		a.spines[["top", "right"]].set_visible(False)
		a.set_ylim(-0.6, len(clusters) - 0.4)
	ax[1].tick_params(left=False)
	ax[2].tick_params(left=False)

	fig.text(0.01, 0.01,
			 "Rows: Leiden clusters of excitatory neurons, labeled with their most common subtype, "
			 "sorted by percent high-pathology cells.\n"
			 "Switching: the cell's GMM label disagrees with its Disease.Group label in more than 10% "
			 "but at most 90% of RecurSieve iterations (GMM 1 matched to high).\n"
		 "Matching labels (green square): the cell's final GMM label equals its own Disease.Group label "
		 "(GMM 1 = high); equal orange and black percentages do not imply this.",
			 fontsize=8, color="#52514e", va="bottom")
	fig.tight_layout(rect=(0, 0.075, 1, 1))
	fig.savefig(args.out_dir / "alzheimers_labels_cluster_plot.pdf")
	fig.savefig(args.out_dir / "alzheimers_labels_cluster_plot.png", dpi=200)
	plt.close(fig)
	print(f"Wrote {args.out_dir / 'alzheimers_labels_cluster_plot.png'}")


if __name__ == "__main__":
	main()
