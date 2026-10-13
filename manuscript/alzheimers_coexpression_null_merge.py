"""Add batches of extra shuffles to a saved co-expression null.

Reads the saved outputs of alzheimers_coexpression_null.py in DIR and every
batch under DIR/null_batches/ (written with --first-shuffle and --batch-dir),
checks that each batch refit the same observed panel, then rewrites the
outputs from all shuffles together. Per-gene p-values get a Benjamini-Hochberg
adjusted column (p_adj) across the observed panel.

Usage:
	python alzheimers_coexpression_null_merge.py --out-dir DIR

Outputs (to DIR, overwritten):
- alzheimers_coexpression_null_runs.csv
- alzheimers_coexpression_null_flip_rates.csv
- alzheimers_coexpression_null_panels.csv
- alzheimers_coexpression_null_gene_frequency.csv (with p_adj)
- alzheimers_coexpression_null_pvalues.csv
- alzheimers_coexpression_null_flip_rates.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import false_discovery_control

import recursieve
from alzheimers_coexpression_null import plot_flip_rates

PREFIX = "alzheimers_coexpression_null_"
TABLES = ("runs", "flip_rates", "panels")


def read_tables(folder: Path) -> dict:
	return {key: pd.read_csv(folder / f"{PREFIX}{key}.csv") for key in TABLES}


def observed_panel(panels: pd.DataFrame) -> list:
	obs = panels[panels["run"] == "observed"].sort_values("rank")
	return obs["gene"].tolist()


def main() -> None:
	plt.rcParams.update({"font.family": "Arial", "font.size": 12,
						 "pdf.fonttype": 42, "ps.fonttype": 42})
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path, required=True)
	parser.add_argument("--flip-threshold", type=float, default=0.01,
						help="Stop threshold drawn on the flip-rate plot")
	args = parser.parse_args()
	out_dir = args.out_dir

	saved = read_tables(out_dir)
	panel = observed_panel(saved["panels"])
	batches = sorted((out_dir / "null_batches").glob("*/"))
	if not batches:
		raise SystemExit(f"No batches in {out_dir / 'null_batches'}")

	merged = {key: [saved[key]] for key in TABLES}
	for batch in batches:
		tables = read_tables(batch)
		if observed_panel(tables["panels"]) != panel:
			raise SystemExit(f"{batch.name}: observed panel differs from the saved one")
		for key in TABLES:
			df = tables[key]
			merged[key].append(df[df["run"] != "observed"])
		print(f"Added {batch.name}")

	# rerunning the merge must not count a shuffle twice
	runs = pd.concat(merged["runs"], ignore_index=True).drop_duplicates("run", keep="first")
	flips = pd.concat(merged["flip_rates"], ignore_index=True).drop_duplicates(["run", "iteration"])
	panels = pd.concat(merged["panels"], ignore_index=True).drop_duplicates(["run", "rank"])

	freq = recursieve.recursieve._null_gene_frequency(panels)
	freq["p_adj"] = false_discovery_control(freq["p_value"], method="bh")
	pvalues = recursieve.recursieve._null_pvalues(runs)

	outputs = {"runs": runs, "flip_rates": flips, "panels": panels,
			   "gene_frequency": freq, "pvalues": pvalues}
	for key, df in outputs.items():
		df.to_csv(out_dir / f"{PREFIX}{key}.csv", index=False)
	plot_flip_rates(flips, args.flip_threshold, out_dir / f"{PREFIX}flip_rates")

	n = runs["run"].ne("observed").sum()
	print(f"{n} shuffles; {int((freq['p_adj'] < 0.05).sum())}/{len(freq)} genes with p_adj < 0.05")
	print(pvalues.round(4).to_string(index=False))


if __name__ == "__main__":
	main()
