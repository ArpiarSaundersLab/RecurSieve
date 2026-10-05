"""Per-gene summary of a recursieve run, built from the other scripts' outputs.

Usage:
	python alzheimers_summary.py [--out-dir DIR]

Inputs (from DIR): alzheimers_genes.csv, alzheimers_de_full.csv,
alzheimers_seed_stability.csv, alzheimers_coexpression_null_gene_frequency.csv,
alzheimers_rf_vs_recursieve_genes.csv, alzheimers_rf_vs_recursieve_panels.csv.
A missing input leaves its columns empty.

Outputs (to DIR):
- alzheimers_recursieve_summary.csv: One row per recursieve gene
- alzheimers_summary_panels.csv: Held-out AUC of each panel, random panels averaged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
	"""Read a CSV, or return an empty frame if it is missing."""
	if path.exists():
		return pd.read_csv(path)
	print(f"Warning: {path.name} not found; its columns will be empty")
	return pd.DataFrame(columns=columns)


def build_summary(out_dir: Path) -> pd.DataFrame:
	"""One row per recursieve gene, in selection order."""
	genes = pd.read_csv(out_dir / "alzheimers_genes.csv")[["gene"]]
	genes.insert(0, "rank", range(1, len(genes) + 1))

	de = _read(out_dir / "alzheimers_de_full.csv", ["gene", "logfc", "pval", "pval_adj", "is_de"])
	seed = _read(out_dir / "alzheimers_seed_stability.csv", ["gene", "seed_frequency"])
	null = _read(out_dir / "alzheimers_coexpression_null_gene_frequency.csv",
				 ["gene", "n_shuffled_panels", "p_value"])
	rf = _read(out_dir / "alzheimers_rf_vs_recursieve_genes.csv",
			   ["panel", "gene", "plain_rf_rank", "in_other_panel"])
	rf = rf[rf["panel"] == "recursieve"] if len(rf) else rf

	summary = (genes
			   .merge(de[["gene", "logfc", "pval", "pval_adj", "is_de"]], on="gene", how="left")
			   .merge(seed[["gene", "seed_frequency"]], on="gene", how="left")
			   .merge(null[["gene", "n_shuffled_panels", "p_value"]]
					  .rename(columns={"n_shuffled_panels": "null_shuffled_panels",
									   "p_value": "null_pval"}), on="gene", how="left")
			   .merge(rf[["gene", "plain_rf_rank", "in_other_panel"]]
					  .rename(columns={"in_other_panel": "in_plain_rf_topN"}), on="gene", how="left"))
	# not in the plain RF top N (N = panel size)
	summary["missed_by_plain_rf"] = summary["in_plain_rf_topN"].map(
		lambda v: pd.NA if pd.isna(v) else not bool(v))
	return summary


def build_panels(out_dir: Path) -> pd.DataFrame:
	"""Panel scores with random panels averaged."""
	panels = _read(out_dir / "alzheimers_rf_vs_recursieve_panels.csv",
				   ["panel", "n_genes", "accuracy", "auc"])
	if panels.empty:
		return panels
	is_random = panels["panel"].str.startswith("random_")
	random = panels[is_random]
	rows = panels[~is_random].assign(accuracy_sd=float("nan"), auc_sd=float("nan"))
	if len(random):
		rows = pd.concat([rows, pd.DataFrame([{
			"panel": f"random_mean_of_{len(random)}", "n_genes": random["n_genes"].iloc[0],
			"accuracy": random["accuracy"].mean(), "auc": random["auc"].mean(),
			"accuracy_sd": random["accuracy"].std(), "auc_sd": random["auc"].std(),
		}])], ignore_index=True)
	return rows[["panel", "n_genes", "accuracy", "accuracy_sd", "auc", "auc_sd"]]


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	out_dir = parser.parse_args().out_dir

	summary = build_summary(out_dir)
	summary.to_csv(out_dir / "alzheimers_recursieve_summary.csv", index=False)
	panels = build_panels(out_dir)
	panels.to_csv(out_dir / "alzheimers_summary_panels.csv", index=False)
	print(summary.round(4).to_string(index=False))
	print("\n" + panels.round(4).to_string(index=False))


if __name__ == "__main__":
	main()
