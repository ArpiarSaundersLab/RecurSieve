"""Plain random forest on the feature comparison multi-wide datasets.

Asks whether a plain RF, with no recursion, ranks any of the staged genes
as important. Uses the same datasets as `feature_comparision.py --datasets 25`
(dataset d uses sim_seed + d and stage_seed + d) by importing its builders.
The RF is fit once per dataset (one trial).

A gene is called important when its RF importance is above --threshold
(default 0.01). With 2000 genes, an even split gives 0.0005 per gene.

RF settings match alzheimers_rf_baseline.py: 300 trees, max_features="sqrt",
class_weight="balanced_subsample", stratified 70/30 split, input is
normalize_total(1e4) + log1p.

Outputs (to --out-dir, new files only):
  rf_test_per_dataset.csv   one row per dataset and scenario
  rf_test_importances.csv   importance and rank of every staged gene

Usage:
  python feature_comparison_rf_test.py
  python feature_comparison_rf_test.py --datasets 5 --scenarios bimodal
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# builders and checks come from the existing benchmark, unchanged
from feature_comparision import BUILDERS, SCENARIO_ORDER, check_staging, _normalized


def run_rf(adata, seed):
	"""Fit an RF on group labels and return importances and test AUC."""
	a = _normalized(adata)
	X = np.asarray(a.X, dtype=np.float32)
	y = (a.obs["group"].astype(str).values == "group_2").astype(int)
	idx_train, idx_test = train_test_split(
		np.arange(a.n_obs), test_size=0.3, random_state=seed, stratify=y)
	rf = RandomForestClassifier(
		n_estimators=300, max_depth=None, max_features="sqrt",
		min_samples_split=2, min_samples_leaf=1,
		class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
	rf.fit(X[idx_train], y[idx_train])
	auc = roc_auc_score(y[idx_test], rf.predict_proba(X[idx_test])[:, 1])
	return pd.Series(rf.feature_importances_, index=a.var_names), auc


def main():
	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--datasets", type=int, default=25)
	ap.add_argument("--threshold", type=float, default=0.01)
	ap.add_argument("--rf-seed", type=int, default=42)
	ap.add_argument("--n-staged", type=int, default=10)
	ap.add_argument("--ncells", type=int, default=2000)
	ap.add_argument("--ngenes", type=int, default=2000)
	ap.add_argument("--sim-seed", type=int, default=32)
	ap.add_argument("--stage-seed", type=int, default=0)
	ap.add_argument("--out-dir", default=str(
		Path(__file__).resolve().parent / "robustness" / "feature_comparison_rf_test"))
	args = ap.parse_args()

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	rows, imp_rows = [], []
	for ds in range(args.datasets):
		for scen in scenarios:
			t0 = time.time()
			adata, staged = BUILDERS[scen](
				sim_seed=args.sim_seed + ds, stage_seed=args.stage_seed + ds,
				n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes)
			check_staging(adata, staged)
			imp, auc = run_rf(adata, args.rf_seed)
			rank = imp.rank(ascending=False, method="min").astype(int)
			above = imp[imp > args.threshold]
			n_staged_above = int(above.index.isin(staged).sum())
			rows.append(dict(
				dataset=ds, scenario=scen, test_auc=auc,
				n_staged_above=n_staged_above,
				n_other_above=len(above) - n_staged_above,
				max_staged_importance=float(imp[staged].max()) if staged else np.nan,
				best_staged_rank=int(rank[staged].min()) if staged else np.nan,
				max_other_importance=float(imp.drop(staged).max()),
			))
			for g in staged:
				imp_rows.append(dict(dataset=ds, scenario=scen, gene=g,
									 importance=float(imp[g]), rank=int(rank[g])))
			r = rows[-1]
			print(f"  [d{ds} {scen}] staged>{args.threshold:g}: {n_staged_above}  "
				  f"other>{args.threshold:g}: {r['n_other_above']}  "
				  f"best staged rank: {r['best_staged_rank']}  AUC {auc:.3f}  "
				  f"[{time.time()-t0:.0f}s]", flush=True)
		# save after each dataset so a crash loses little
		pd.DataFrame(rows).to_csv(out_dir / "rf_test_per_dataset.csv", index=False)
		pd.DataFrame(imp_rows).to_csv(out_dir / "rf_test_importances.csv", index=False)

	df = pd.DataFrame(rows)
	order = [s for s in SCENARIO_ORDER if s in scenarios]
	summary = df.groupby("scenario").agg(
		mean_staged_above=("n_staged_above", "mean"),
		datasets_with_any=("n_staged_above", lambda v: int((v > 0).sum())),
		mean_other_above=("n_other_above", "mean"),
		median_best_staged_rank=("best_staged_rank", "median"),
		mean_test_auc=("test_auc", "mean"),
	).reindex(order)
	print(f"\nSummary over {args.datasets} datasets (threshold {args.threshold:g})")
	print(summary.round(3).to_string())


if __name__ == "__main__":
	main()
