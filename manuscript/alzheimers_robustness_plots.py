"""Seed stability plot for the recursieve panel.

Reads alzheimers_genes.csv, alzheimers_seed_gene_frequency.csv, and
alzheimers_seed_runs.csv.

Usage:
	python alzheimers_robustness_plots.py [--out-dir DIR]

Outputs (to DIR):
- alzheimers_seed_stability.csv: Seed frequency of each panel gene
- alzheimers_seed_stability.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
RECURSIEVE = "#4a3aa7"

# reference line at half of the seeds
STABLE_SEED = 0.5


def style_axis(ax) -> None:
	"""Axis style."""
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)
	ax.set_axisbelow(True)


def plot_seed_stability(table: pd.DataFrame, n_seeds: int, out_stem: Path) -> None:
	"""Bars of seed frequency per gene, most stable at the top."""
	t = table.sort_values(["seed_frequency", "rf_rank"],
						  ascending=[True, False]).reset_index(drop=True)
	y = np.arange(len(t))

	fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(t) + 1.8), dpi=300,
						   layout="constrained")
	ax.barh(y, t["seed_frequency"], height=0.7, color=RECURSIEVE,
			edgecolor="white", linewidth=1.0)
	ax.axvline(STABLE_SEED, color=INK_2, lw=1.0, ls="--")
	ax.set_xlim(0, 1)
	ax.set_yticks(y)
	ax.set_yticklabels(t["gene"], fontsize=11, color=INK)
	ax.set_ylim(-0.6, len(t) - 0.4)
	ax.set_xlabel("Fraction of seeds selecting the gene", fontsize=13, color=INK)
	ax.set_title(f"Seed stability (n = {n_seeds} seeds)", fontsize=14,
				 fontweight="bold", color=INK, loc="left")
	ax.grid(axis="x", color=GRID, lw=0.8)
	style_axis(ax)
	fig.supxlabel(f"Dashed line: selected in {STABLE_SEED:.0%} of seeds.",
				  fontsize=10, color=INK_2)
	for ext in ("png", "pdf"):
		fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	plt.rcParams.update({
		"font.family": "Arial", "font.size": 12,
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	out_dir = parser.parse_args().out_dir

	genes = pd.read_csv(out_dir / "alzheimers_genes.csv")["gene"].tolist()
	seed_freq = pd.read_csv(out_dir / "alzheimers_seed_gene_frequency.csv")
	n_seeds = len(pd.read_csv(out_dir / "alzheimers_seed_runs.csv"))

	freq = dict(zip(seed_freq["gene"], seed_freq["frequency"]))
	table = pd.DataFrame({
		"gene": genes,
		"rf_rank": range(len(genes)),
		"seed_frequency": [freq.get(g, 0.0) for g in genes],
	})
	table.to_csv(out_dir / "alzheimers_seed_stability.csv", index=False)

	plot_seed_stability(table, n_seeds, out_dir / "alzheimers_seed_stability")
	print(f"Genes selected in >= {STABLE_SEED:.0%} of seeds: "
		  f"{int((table['seed_frequency'] >= STABLE_SEED).sum())} / {len(table)}")


if __name__ == "__main__":
	main()
