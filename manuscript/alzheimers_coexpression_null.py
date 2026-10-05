"""Co-expression null for recursieve on the Alzheimer's dataset.

Fits recursieve on the real data, then reruns it on datasets where every raw
count is drawn from the pooled counts of genes and labels are permuted.

Usage:
	python alzheimers_coexpression_null.py [--out-dir DIR] [--n-shuffles N]
		[--n-parallel N] [--cell-type TYPE] [--n-cells N] [--balance-groups]

Outputs (to DIR):
- alzheimers_coexpression_null_runs.csv: Real and shuffled run summaries
- alzheimers_coexpression_null_flip_rates.csv: Flip rate per iteration
- alzheimers_coexpression_null_panels.csv: Gene panel of every run
- alzheimers_coexpression_null_gene_frequency.csv: Per-gene p-values
- alzheimers_coexpression_null_pvalues.csv: Module-level p-values
- alzheimers_coexpression_null_flip_rates.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import recursieve
from alz_loading import add_loading_args, load_alzheimers

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
RECURSIEVE = "#4a3aa7"
CONTEXT = "#8f8e89"


def plot_flip_rates(curves: pd.DataFrame, threshold: float, out_stem: Path) -> None:
	"""Flip rate per iteration for the real and shuffled runs."""
	fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300, layout="constrained")
	shuffled = curves[curves["run"] != "observed"]
	n_shuffles = shuffled["run"].nunique()
	# label the shuffled runs once, at the run that ends last
	label_run = shuffled.loc[shuffled["iteration"].idxmax(), "run"] if n_shuffles else None
	for name, df in curves.groupby("run", sort=False):
		observed = name == "observed"
		ax.plot(df["iteration"], df["flip_rate"], lw=2.5 if observed else 1.5,
				color=RECURSIEVE if observed else CONTEXT, zorder=3 if observed else 2)
		if observed or name == label_run:
			last = df.iloc[-1]
			ax.annotate("observed" if observed else f"shuffled genes (n = {n_shuffles})",
						xy=(last["iteration"], last["flip_rate"]), xytext=(6, 0),
						textcoords="offset points", va="center", fontsize=11,
						color=INK if observed else INK_2)
	ax.axhline(threshold, color=INK_2, lw=1.0, ls="--")
	ax.annotate(f"stop threshold ({threshold:g})", xy=(0, threshold), xycoords=("axes fraction", "data"),
				xytext=(4, 4), textcoords="offset points", ha="left", fontsize=10, color=INK_2)
	ax.set_yscale("log")
	ax.set_xlabel("Iteration", fontsize=13, color=INK)
	ax.set_ylabel("Label flip rate vs previous iteration", fontsize=13, color=INK)
	ax.set_title("Convergence on real vs co-expression-shuffled data", fontsize=14,
				 fontweight="bold", color=INK, loc="left")
	ax.grid(axis="y", color=GRID, lw=0.8)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)
	for ext in ("png", "pdf"):
		fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	plt.rcParams.update({"font.family": "Arial", "font.size": 12,
						 "pdf.fonttype": 42, "ps.fonttype": 42})
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	parser.add_argument("--n-shuffles", type=int, default=25)
	parser.add_argument("--n-parallel", type=int, default=None,
						help="Shuffles run at once (default: about one per 4 cores)")
	add_loading_args(parser)
	args = parser.parse_args()
	out_dir = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	adata_path = Path(__file__).resolve().parent.parent / "data" / "alz.h5ad"
	adata = load_alzheimers(adata_path, args.cell_type, args.cell_type_key,
							args.n_cells, args.balance_groups)
	model = recursieve.recursieve(
		adata=adata, group1="low", group2="high", field_name="Disease.Group",
		plots=False, print_to_console=False, max_iterations=50,
	)

	print(f"Running co-expression null ({args.n_shuffles} shuffles)...")
	res = model.coexpression_null(n_shuffles=args.n_shuffles, n_parallel=args.n_parallel,
								  verbose=10)
	for key in ("runs", "flip_rates", "panels", "gene_frequency", "pvalues"):
		res[key].to_csv(out_dir / f"alzheimers_coexpression_null_{key}.csv", index=False)
	plot_flip_rates(res["flip_rates"], model.flip_rate_percentage,
					out_dir / "alzheimers_coexpression_null_flip_rates")
	print(res["pvalues"].round(4).to_string(index=False))


if __name__ == "__main__":
	main()
