"""Add co-expression null shuffles to the Alzheimer's robustness results (HPC).

Run from inside the robustness/ directory. Each group folder (microglia,
astrocytes, inhibitory, excitatory, all_cell_types) already holds the saved
null outputs; this script adds more shuffles without repeating old ones.

Run mode (one array task):
  Loads the group's cells exactly as the original run did and stops if they
  differ from the original cells. By default it then refits the observed panel
  and stops if it differs from the saved one. With --use-saved-panel it skips
  the refit: it checks that preprocessing kept the same 2000 genes as the
  original run (alzheimers_de_full.csv) and takes the panel length and genes
  from the saved results. The shuffles only use the panel's length and gene
  list, so this gives the same null as a matching refit. It then runs shuffles
  FIRST..FIRST+N-1. Shuffle k uses seed (42, k), the same scheme as
  recursieve.coexpression_null, so new shuffles never repeat old ones. The
  batch is written to <group>/null_batches/shuffles_<FIRST>-<LAST>/ and a
  finished batch is skipped on a rerun.

Merge mode (--merge):
  For each group, combines the saved shuffles with every batch, recomputes the
  per-gene p-values (with a Benjamini-Hochberg p_adj column) and the
  module-level p-values, and overwrites the group's CSVs.

Usage:
  python alzheimers_null_hpc.py --group microglia --first-shuffle 51 --n-shuffles 5 \\
      --data /path/to/alz.h5ad
  python alzheimers_null_hpc.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control

import recursieve

PREFIX = "alzheimers_coexpression_null_"
TABLES = ("runs", "flip_rates", "panels")
FIELD_NAME = "Disease.Group"
GROUPS = ("low", "high")
ALL_GROUPS = ["microglia", "astrocytes", "inhibitory", "excitatory", "all_cell_types"]
# cell selection used for each group's original run
SELECTION = {
	"microglia": dict(cell_type="Microglia", n_cells=3964, balance_groups=True),
	"astrocytes": dict(cell_type="Astrocytes", n_cells=7818, balance_groups=True),
	"inhibitory": dict(cell_type="Inhibitory", n_cells=25000, balance_groups=True),
	"excitatory": dict(cell_type="Excitatory", n_cells=25000, balance_groups=True),
	"all_cell_types": dict(cell_type=None, n_cells=50000, balance_groups=False),
}


# fingerprint of each group's cell list in the original run (first 16 hex
# characters of sha256 over the selected cell names, one per line)
CELL_HASH = {
	"microglia": "ef3e999a9f0246c5",
	"astrocytes": "26d8d8162ef07897",
	"inhibitory": "9643e3f9e2bee0a8",
	"excitatory": "31645efb3cb7ef3d",
	"all_cell_types": "324b656ffb064b25",
}


def select_cells(obs, cell_type=None, cell_type_key="Major_celltypes", n_cells=50000,
				 balance_groups=False, seed=0):
	"""Row positions of the chosen cells, in output order (same as alz_loading.py)."""
	pos = np.arange(len(obs))
	if cell_type is not None:
		pos = pos[(obs[cell_type_key] == cell_type).to_numpy()]
		print(f"Kept {cell_type} cells only: {len(pos)}")
	sub = obs.iloc[pos]

	if balance_groups:
		rng = np.random.default_rng(seed)
		keep = []
		for g in GROUPS:
			cells = np.flatnonzero((sub[FIELD_NAME] == g).to_numpy())
			keep.append(rng.choice(cells, size=min(n_cells // 2, len(cells)), replace=False))
			print(f"Sampled {len(keep[-1])} {g} cells")
		return pos[np.sort(np.concatenate(keep))]

	# same cells and order as sc.pp.subsample on the full data
	light = ad.AnnData(obs=sub.reset_index(drop=True).assign(_row=np.arange(len(sub))))
	light = sc.pp.subsample(light, n_obs=min(n_cells, len(sub)), random_state=seed, copy=True)
	return pos[light.obs["_row"].to_numpy()]


def load_alzheimers(path, **selection):
	"""Load the chosen cells with gene symbols as var_names (same as alz_loading.py)."""
	print(f"Loading data from: {path} (backed)")
	backed = ad.read_h5ad(path, backed="r")
	order = select_cells(backed.obs, **selection)
	# backed indexing needs sorted rows
	rows = np.sort(order)
	adata = backed[rows].to_memory()
	backed.file.close()
	adata = adata[np.searchsorted(rows, order)].copy()
	# plain strings: some anndata/pandas versions keep this column categorical,
	# and making duplicate names unique then fails
	adata.var_names = adata.var["feature_name"].astype(str).to_numpy()
	adata.var_names_make_unique()
	return adata


def observed_panel(panels):
	obs = panels[panels["run"] == "observed"].sort_values("rank")
	return obs["gene"].tolist()


def read_tables(folder):
	return {key: pd.read_csv(folder / f"{PREFIX}{key}.csv") for key in TABLES}


def run_batch(args):
	group_dir = Path(args.group)
	last = args.first_shuffle + args.n_shuffles - 1
	batch = group_dir / "null_batches" / f"shuffles_{args.first_shuffle}-{last}"
	if (batch / f"{PREFIX}panels.csv").exists():
		print(f"{batch} already done, skipping")
		return
	saved = read_tables(group_dir)
	saved_panel = observed_panel(saved["panels"])

	adata = load_alzheimers(args.data, **SELECTION[args.group])
	cell_hash = hashlib.sha256("\n".join(adata.obs_names).encode()).hexdigest()[:16]
	if cell_hash != CELL_HASH[args.group]:
		raise SystemExit(f"{args.group}: selected cells differ from the original run "
						 f"({cell_hash} vs {CELL_HASH[args.group]})")
	print(f"{args.group}: cells match the original run ({adata.n_obs} cells)")

	model_args = dict(adata=adata, group1="low", group2="high", field_name=FIELD_NAME,
					  plots=False, print_to_console=False, max_iterations=50)
	if args.use_saved_panel:
		# preprocess only; the panel comes from the saved results
		model = recursieve.recursieve(**model_args, run=False)
		model.preprocessing()
		model.filter_groups()
		genes = set(pd.read_csv(group_dir / "alzheimers_de_full.csv")["gene"])
		if set(model.adata.var_names) != genes:
			raise SystemExit(f"{args.group}: preprocessing kept different genes than "
							 "the original run (alzheimers_de_full.csv)")
		print(f"{args.group}: preprocessed genes match the original run; "
			  f"using the saved panel ({len(saved_panel)} genes)")
		panel = saved_panel
	else:
		model = recursieve.recursieve(**model_args)
		if list(model.genes) != saved_panel:
			raise SystemExit(f"{args.group}: observed panel differs from the saved one; "
							 "rerun with --use-saved-panel")
		print(f"{args.group}: observed panel matches the saved one ({len(saved_panel)} genes)")
		panel = list(model.genes)

	# same setup as recursieve.coexpression_null: shuffled panels match the
	# observed length, with stopping rules off
	X = model._get_X_dense(ad.AnnData(X=model.adata.layers["counts"]))
	obs = model.adata.obs[[model.field_name]].copy()
	params = dict(model._params, plots=False, print_to_console=False,
				  preprocess=False, run=False, flip_rate_percentage=-1.0,
				  patience=None, max_iterations=len(panel) - 1)
	cores = len(os.sched_getaffinity(0))
	n_parallel = max(1, min(args.n_parallel, args.n_shuffles))
	params["n_jobs"] = max(1, cores // n_parallel)
	shuffles = range(args.first_shuffle, last + 1)
	print(f"Running shuffles {args.first_shuffle}-{last}, {n_parallel} at a time "
		  f"on {cores} cores", flush=True)
	results = Parallel(n_jobs=n_parallel, verbose=10)(
		delayed(recursieve.recursieve._run_null_shuffle)(
			X, obs, model.adata.var_names, params, (model.seed, k), panel)
		for k in shuffles
	)

	# the observed run's rows, as saved by the original run
	rows = saved["runs"][saved["runs"]["run"] == "observed"].to_dict("records")
	curves = [tuple(r) for r in saved["flip_rates"][saved["flip_rates"]["run"] == "observed"]
			  [["run", "iteration", "flip_rate"]].itertuples(index=False)]
	panels = [("observed", i + 1, g) for i, g in enumerate(panel)]
	for k, (summary, flips, genes) in zip(shuffles, results):
		name = f"shuffle_{k}"
		rows.append({"run": name, **summary})
		curves += [(name, i + 2, f) for i, f in enumerate(flips)]
		panels += [(name, i + 1, g) for i, g in enumerate(genes)]
	batch.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows).to_csv(batch / f"{PREFIX}runs.csv", index=False)
	pd.DataFrame(curves, columns=["run", "iteration", "flip_rate"]).to_csv(
		batch / f"{PREFIX}flip_rates.csv", index=False)
	# panels last: its presence marks the batch as finished
	pd.DataFrame(panels, columns=["run", "rank", "gene"]).to_csv(
		batch / f"{PREFIX}panels.csv", index=False)
	print(f"Batch written to {batch}")


def merge_group(group):
	group_dir = Path(group)
	saved = read_tables(group_dir)
	panel = observed_panel(saved["panels"])
	batches = sorted(b for b in (group_dir / "null_batches").glob("*/")
					 if (b / f"{PREFIX}panels.csv").exists())
	merged = {key: [saved[key]] for key in TABLES}
	for batch in batches:
		tables = read_tables(batch)
		if observed_panel(tables["panels"]) != panel:
			raise SystemExit(f"{group}/{batch.name}: observed panel differs from the saved one")
		for key in TABLES:
			df = tables[key]
			merged[key].append(df[df["run"] != "observed"])

	# a shuffle already in the saved files is not counted twice
	runs = pd.concat(merged["runs"], ignore_index=True).drop_duplicates("run", keep="first")
	flips = pd.concat(merged["flip_rates"], ignore_index=True).drop_duplicates(["run", "iteration"])
	panels = pd.concat(merged["panels"], ignore_index=True).drop_duplicates(["run", "rank"])

	freq = recursieve.recursieve._null_gene_frequency(panels)
	freq["p_adj"] = false_discovery_control(freq["p_value"], method="bh")
	pvalues = recursieve.recursieve._null_pvalues(runs)
	outputs = {"runs": runs, "flip_rates": flips, "panels": panels,
			   "gene_frequency": freq, "pvalues": pvalues}
	for key, df in outputs.items():
		df.to_csv(group_dir / f"{PREFIX}{key}.csv", index=False)
	n = int(runs["run"].ne("observed").sum())
	print(f"{group}: {n} shuffles, {int((freq['p_adj'] < 0.05).sum())}/{len(freq)} genes "
		  f"with p_adj < 0.05")


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--group", choices=ALL_GROUPS)
	ap.add_argument("--first-shuffle", type=int)
	ap.add_argument("--n-shuffles", type=int, default=5)
	ap.add_argument("--n-parallel", type=int, default=2,
					help="shuffles run at once; each gets cores / n-parallel cores")
	ap.add_argument("--data", help="path to alz.h5ad")
	ap.add_argument("--use-saved-panel", action="store_true",
					help="skip refitting the observed panel; use the saved one")
	ap.add_argument("--merge", action="store_true",
					help="combine the saved shuffles with every batch, for all groups")
	args = ap.parse_args()

	if args.merge:
		for group in ALL_GROUPS:
			merge_group(group)
		return
	if not (args.group and args.first_shuffle and args.data):
		ap.error("run mode needs --group, --first-shuffle and --data")
	run_batch(args)


if __name__ == "__main__":
	main()
