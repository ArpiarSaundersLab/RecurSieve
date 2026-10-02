"""Compare recursieve gene selection against a plain Random Forest baseline.

This script:
1. Loads and subsamples the Alzheimer's dataset exactly as in
   alzheimers_analysis.py (50,000 cells, same subsample seed)
2. Applies recursieve's own preprocessing and group filtering so the cells and
   2,000 HVGs are identical to the recursieve run
3. Trains a single Random Forest (same hyperparameters, seed, and 70/30 split)
   on the true Disease.Group labels ("low" vs "high") and ranks all genes by
   feature importance
4. Compares the top-N RF genes (N = size of the recursieve panel) against the
   recursieve genes saved in alzheimers_genes.csv:
   - overlap, Jaccard index, and hypergeometric enrichment of the overlap
   - where each recursieve gene falls in the plain RF importance ranking
   - fraction of each panel that is significantly DE
   - held-out low vs high classification using only each panel's genes,
     alongside random gene panels of the same size

Outputs are written to the current directory (expected: manuscript/):
- alzheimers_rf_baseline_ranking.csv: Full plain RF importance ranking
- alzheimers_rf_vs_recursieve_genes.csv: Per-gene comparison of both panels
- alzheimers_rf_vs_recursieve_panels.csv: Panel-level classification metrics
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import recursieve

GROUP1 = "low"
GROUP2 = "high"
FIELD_NAME = "Disease.Group"
N_CELLS = 50000
N_RANDOM_PANELS = 10


def prepare_adata(adata: ad.AnnData) -> recursieve.recursieve:
	"""Run recursieve preprocessing, group filtering, and DE without selection.

	Builds a recursieve instance without calling ``__init__`` so only the
	data-preparation steps run, guaranteeing identical cells and genes to the
	full recursieve analysis.
	"""
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
		"max_depth": None,
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
	"""Fit an RF and return the model, accuracy, and AUC for GROUP2."""
	rf = RandomForestClassifier(**rf_params, random_state=seed)
	rf.fit(X_train, y_train)
	acc = accuracy_score(y_test, rf.predict(X_test))
	proba = rf.predict_proba(X_test)[:, list(rf.classes_).index(GROUP2)]
	auc = roc_auc_score((y_test == GROUP2).astype(int), proba)
	return rf, acc, auc


def main() -> None:
	"""Train a plain RF baseline and compare it to the recursieve results."""
	data_dir = Path(__file__).resolve().parent.parent / "data"
	out_dir = Path(__file__).resolve().parent
	adata_path = data_dir / "alz.h5ad"
	recursieve_genes_path = out_dir / "alzheimers_genes.csv"

	if not adata_path.exists():
		raise FileNotFoundError(f"Data file not found: {adata_path}")
	if not recursieve_genes_path.exists():
		raise FileNotFoundError(f"Recursieve results not found: {recursieve_genes_path}")

	rs_genes = pd.read_csv(recursieve_genes_path)["gene"].tolist()
	n_panel = len(rs_genes)

	# Same loading and subsampling as alzheimers_analysis.py
	print(f"Loading data from: {adata_path}")
	adata_alz = ad.read_h5ad(adata_path)
	adata_alz = sc.pp.subsample(adata_alz, n_obs=N_CELLS, copy=True)
	adata_alz.var_names = adata_alz.var['feature_name']
	adata_alz.var_names_make_unique()
	print(f"Data shape: {adata_alz.n_obs} cells x {adata_alz.n_vars} genes")

	model = prepare_adata(adata_alz)
	del adata_alz
	adata = model.adata
	print(f"After preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

	missing = [g for g in rs_genes if g not in adata.var_names]
	if missing:
		raise ValueError(f"Recursieve genes missing from preprocessed data: {missing}")

	# Plain RF on true labels, same split and parameters as recursieve's first fit
	X = model._get_X_dense()
	y = adata.obs[FIELD_NAME].astype(str).values
	idx_train, idx_test = train_test_split(
		np.arange(adata.n_obs), test_size=0.3, random_state=model.seed, stratify=y
	)
	print("\nTraining plain Random Forest on all genes...")
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

	# Overlap statistics (universe = all genes after preprocessing)
	n_universe = adata.n_vars
	overlap = [g for g in rs_genes if g in set(rf_genes)]
	jaccard = len(overlap) / len(set(rs_genes) | set(rf_genes))
	expected = n_panel * n_panel / n_universe
	hyper_p = hypergeom.sf(len(overlap) - 1, n_universe, n_panel, n_panel)

	rs_ranks = np.array([rank_of[g] for g in rs_genes])
	de_set = set(model.de_dict.keys())
	rs_de_frac = np.mean([g in de_set for g in rs_genes])
	rf_de_frac = np.mean([g in de_set for g in rf_genes])

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

	# Held-out classification of low vs high using only each panel's genes
	print("\nScoring gene panels on held-out cells...")
	col = {g: i for i, g in enumerate(adata.var_names)}

	def score_panel(genes):
		cols = [col[g] for g in genes]
		_, acc, auc = fit_and_score(
			X[idx_train][:, cols], X[idx_test][:, cols], y[idx_train], y[idx_test],
			model.rf_params, model.seed,
		)
		return acc, auc

	panel_rows = [{"panel": "all_genes", "n_genes": n_universe,
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
	random_panels = panels[panels["panel"].str.startswith("random_")]

	# Report
	print("\n" + "=" * 70)
	print("RECURSIEVE vs PLAIN RANDOM FOREST (Alzheimer's, low vs high)")
	print("=" * 70)
	print(f"Cells x genes: {adata.n_obs} x {n_universe}   Panel size N = {n_panel}")
	print(f"Plain RF top gene: {rf_genes[0]}   Recursieve first gene: {rs_genes[0]}")
	print(f"\nOverlap of top-{n_panel}: {len(overlap)} / {n_panel} "
		  f"(Jaccard = {jaccard:.3f}, expected by chance = {expected:.2f}, "
		  f"hypergeometric p = {hyper_p:.2e})")
	print(f"Shared genes: {', '.join(overlap)}")
	print(f"Recursieve-only: {', '.join(g for g in rs_genes if g not in set(rf_genes))}")
	print(f"Plain-RF-only: {', '.join(g for g in rf_genes if g not in set(rs_genes))}")
	print(f"\nRecursieve genes in plain RF ranking (of {n_universe}): "
		  f"median rank = {np.median(rs_ranks):.0f}, "
		  f"min = {rs_ranks.min()}, max = {rs_ranks.max()}")
	for k in (50, 100, 200, 500):
		print(f"  in plain RF top {k:>3}: {(rs_ranks <= k).sum()} / {n_panel}")
	print(f"\nFraction DE-significant: recursieve = {rs_de_frac:.2f}, "
		  f"plain RF = {rf_de_frac:.2f} (all genes = {len(de_set) / n_universe:.2f})")
	print("\nHeld-out low vs high classification:")
	for _, r in panels[~panels["panel"].str.startswith("random_")].iterrows():
		print(f"  {r['panel']:<15} n={r['n_genes']:<5} acc={r['accuracy']:.4f}  auc={r['auc']:.4f}")
	print(f"  {'random (x' + str(N_RANDOM_PANELS) + ')':<15} n={n_panel:<5} "
		  f"acc={random_panels['accuracy'].mean():.4f}±{random_panels['accuracy'].std():.4f}  "
		  f"auc={random_panels['auc'].mean():.4f}±{random_panels['auc'].std():.4f}")


if __name__ == "__main__":
	main()
