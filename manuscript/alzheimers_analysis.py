"""Run recursieve on the Alzheimer's dataset (low vs high pathology).

Usage:
	python alzheimers_analysis.py [--out-dir DIR] [--n-seeds N] [--n-parallel N]
		[--cell-type TYPE] [--n-cells N] [--balance-groups]

Outputs (to DIR):
- alzheimers_genes.csv: Selected genes with accuracy, AUC, and OOB score
- alzheimers_unique_genes.csv: Selected genes that are not DE
- alzheimers_de_results.csv: Selected genes that are DE
- alzheimers_de_full.csv: Wilcoxon results for all genes
- alzheimers_seed_runs.csv, alzheimers_seed_gene_frequency.csv,
  alzheimers_seed_panels.csv: Seed stability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

import recursieve
from alz_loading import add_loading_args, load_alzheimers


def de_table(model) -> pd.DataFrame:
	"""Wilcoxon results for all genes, with the DE flag used by recursieve."""
	df = sc.get.rank_genes_groups_df(model.adata, group=model.group2)
	df = df.rename(columns={
		"names": "gene", "logfoldchanges": "logfc", "pvals": "pval",
		"pvals_adj": "pval_adj", "scores": "score",
	})
	df["is_de"] = df["gene"].isin(model.de_dict.keys())
	return df


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	parser.add_argument("--n-seeds", type=int, default=10)
	parser.add_argument("--n-parallel", type=int, default=None,
						help="Seeds run at once (default: about one per 4 cores)")
	add_loading_args(parser)
	args = parser.parse_args()

	out_dir = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)
	adata_path = Path(__file__).resolve().parent.parent / "data" / "alz.h5ad"
	adata = load_alzheimers(adata_path, args.cell_type, args.cell_type_key,
							args.n_cells, args.balance_groups)
	print(f"Data shape: {adata.n_obs} cells x {adata.n_vars} genes")

	model = recursieve.recursieve(
		adata=adata, group1="low", group2="high", field_name="Disease.Group",
		plots=False, print_to_console=False, max_iterations=50,
	)
	n = len(model.genes)
	print(f"Selected {n} genes")

	pd.DataFrame({
		"gene": model.genes,
		"accuracy": model.accuracy_scores[:n],
		"auc": model.auc_scores[:n],
		"oob": model.oob_scores[:n],
	}).to_csv(out_dir / "alzheimers_genes.csv", index=False)
	pd.DataFrame({"gene": model.unique_gene_panel}).to_csv(
		out_dir / "alzheimers_unique_genes.csv", index=False)
	_, de_results = model.intersect_genes_and_de(top_n_genes=n)
	de_results.to_csv(out_dir / "alzheimers_de_results.csv", index=False)
	de_table(model).to_csv(out_dir / "alzheimers_de_full.csv", index=False)

	print(f"Running seed stability ({args.n_seeds} seeds)...")
	seeds = model.seed_stability(n_seeds=args.n_seeds, n_parallel=args.n_parallel)
	seeds["runs"].to_csv(out_dir / "alzheimers_seed_runs.csv", index=False)
	seeds["gene_frequency"].to_csv(out_dir / "alzheimers_seed_gene_frequency.csv", index=False)
	seeds["panels"].to_csv(out_dir / "alzheimers_seed_panels.csv", index=False)
	print(f"Mean pairwise Jaccard across seeds: {seeds['mean_pairwise_jaccard']:.3f}")


if __name__ == "__main__":
	main()
