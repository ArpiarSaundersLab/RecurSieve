"""Compare recursieve genes with a plain random forest baseline.

Trains one RF on all genes with the true low vs high labels, ranks genes by
importance, and scores the recursieve panel, the plain RF top N panel, and
random panels on held-out cells. Cell selection options must match
alzheimers_analysis.py.

Usage:
	python alzheimers_rf_baseline.py [--out-dir DIR] [--max-depth N]
		[--cell-type TYPE] [--n-cells N] [--balance-groups]

Outputs (to DIR):
- alzheimers_rf_baseline_ranking.csv: Plain RF importance ranking
- alzheimers_rf_vs_recursieve_genes.csv: Per-gene comparison of both panels
- alzheimers_rf_vs_recursieve_panels.csv: Held-out accuracy and AUC per panel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import recursieve
from alz_loading import add_loading_args, load_alzheimers

GROUP1 = "low"
GROUP2 = "high"
FIELD_NAME = "Disease.Group"
N_RANDOM_PANELS = 10


def prepare_adata(adata: ad.AnnData, max_depth=None) -> recursieve.recursieve:
	"""Run recursieve preprocessing and DE without gene selection."""
	model = recursieve.recursieve.__new__(recursieve.recursieve)
	model.adata = adata
	model.annsql_db = None
	model.group1 = GROUP1
	model.group2 = GROUP2
	model.field_name = FIELD_NAME
	model.pval_cutoff = 0.05
	model.logfc_cutoff = 0.1
	model.seed = 42
	model.rf_params = {
		"n_estimators": 300,
		"max_depth": max_depth,
		"max_features": "sqrt",
		"min_samples_split": 2,
		"min_samples_leaf": 1,
		"class_weight": "balanced_subsample",
		"n_jobs": -1,
	}
	model.preprocessing()
	model.filter_groups()
	model.scanpy_de_original_groups()
	return model


def fit_and_score(X_train, X_test, y_train, y_test, rf_params, seed):
	"""Fit an RF and return it with test accuracy and AUC."""
	rf = RandomForestClassifier(**rf_params, random_state=seed)
	rf.fit(X_train, y_train)
	acc = accuracy_score(y_test, rf.predict(X_test))
	proba = rf.predict_proba(X_test)[:, list(rf.classes_).index(GROUP2)]
	auc = roc_auc_score((y_test == GROUP2).astype(int), proba)
	return rf, acc, auc


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	parser.add_argument("--max-depth", type=int, default=None,
						help="RF max_depth, match the recursieve run")
	add_loading_args(parser)
	args = parser.parse_args()
	out_dir = args.out_dir

	rs_genes = pd.read_csv(out_dir / "alzheimers_genes.csv")["gene"].tolist()
	n_panel = len(rs_genes)
	adata_path = Path(__file__).resolve().parent.parent / "data" / "alz.h5ad"
	adata_alz = load_alzheimers(adata_path, args.cell_type, args.cell_type_key,
								args.n_cells, args.balance_groups)
	model = prepare_adata(adata_alz, max_depth=args.max_depth)
	del adata_alz
	adata = model.adata

	# same split and parameters as the first recursieve fit
	X = model._get_X_dense()
	y = adata.obs[FIELD_NAME].astype(str).values
	idx_train, idx_test = train_test_split(
		np.arange(adata.n_obs), test_size=0.3, random_state=model.seed, stratify=y
	)
	rf, acc_all, auc_all = fit_and_score(
		X[idx_train], X[idx_test], y[idx_train], y[idx_test],
		model.rf_params, model.seed,
	)

	ranking = pd.DataFrame({
		"gene": adata.var_names,
		"importance": rf.feature_importances_,
	}).sort_values("importance", ascending=False).reset_index(drop=True)
	ranking["rf_rank"] = np.arange(1, len(ranking) + 1)
	ranking["is_de"] = ranking["gene"].isin(model.de_dict.keys())
	ranking["in_recursieve"] = ranking["gene"].isin(rs_genes)
	ranking.to_csv(out_dir / "alzheimers_rf_baseline_ranking.csv", index=False)

	rf_genes = ranking["gene"].head(n_panel).tolist()
	rank_of = dict(zip(ranking["gene"], ranking["rf_rank"]))
	de_set = set(model.de_dict.keys())

	gene_rows = []
	for i, g in enumerate(rs_genes, start=1):
		gene_rows.append({"panel": "recursieve", "gene": g, "panel_rank": i,
						  "plain_rf_rank": rank_of[g], "is_de": g in de_set,
						  "in_other_panel": g in set(rf_genes)})
	for i, g in enumerate(rf_genes, start=1):
		gene_rows.append({"panel": "plain_rf", "gene": g, "panel_rank": i,
						  "plain_rf_rank": rank_of[g], "is_de": g in de_set,
						  "in_other_panel": g in set(rs_genes)})
	pd.DataFrame(gene_rows).to_csv(out_dir / "alzheimers_rf_vs_recursieve_genes.csv", index=False)

	# held-out low vs high classification with each panel's genes
	col = {g: i for i, g in enumerate(adata.var_names)}

	def score_panel(genes):
		cols = [col[g] for g in genes]
		_, acc, auc = fit_and_score(
			X[idx_train][:, cols], X[idx_test][:, cols], y[idx_train], y[idx_test],
			model.rf_params, model.seed,
		)
		return acc, auc

	panel_rows = [{"panel": "all_genes", "n_genes": adata.n_vars,
				   "accuracy": acc_all, "auc": auc_all}]
	for name, genes in [("recursieve", rs_genes), ("plain_rf_topN", rf_genes)]:
		acc, auc = score_panel(genes)
		panel_rows.append({"panel": name, "n_genes": len(genes), "accuracy": acc, "auc": auc})
	rng = np.random.default_rng(model.seed)
	for i in range(N_RANDOM_PANELS):
		genes = rng.choice(np.asarray(adata.var_names), size=n_panel, replace=False).tolist()
		acc, auc = score_panel(genes)
		panel_rows.append({"panel": f"random_{i + 1}", "n_genes": n_panel, "accuracy": acc, "auc": auc})
	panels = pd.DataFrame(panel_rows)
	panels.to_csv(out_dir / "alzheimers_rf_vs_recursieve_panels.csv", index=False)
	print(panels.round(4).to_string(index=False))


if __name__ == "__main__":
	main()
