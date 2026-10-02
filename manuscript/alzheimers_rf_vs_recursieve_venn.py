"""Venn diagram of recursieve vs plain Random Forest gene panels.

Reads the per-gene comparison written by alzheimers_rf_baseline.py and draws a
two-set Venn (styled to match alzheimers_venn.png) with the genes in each
region listed beneath it. Each region carries boxed cards splitting its genes
by DE status (DE, failing the logFC cutoff, or failing the p-value cutoff),
matching alzheimers_venn.png.

Outputs are written to the current directory (expected: manuscript/):
- alzheimers_rf_vs_recursieve_venn.png / .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

INK = "#0b0b0b"
INK_2 = "#52514e"
RECURSIEVE = "#4a3aa7"
PLAIN_RF = "#1baf7a"

PVAL_CUTOFF = 0.05


def main() -> None:
	"""Draw the recursieve vs plain RF Venn diagram with region gene lists."""
	plt.rcParams.update({
		"font.family": "Arial", "font.size": 13,
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	out_dir = Path(__file__).resolve().parent
	df = pd.read_csv(out_dir / "alzheimers_rf_vs_recursieve_genes.csv")

	rec = df[df["panel"] == "recursieve"].sort_values("panel_rank")
	rf = df[df["panel"] == "plain_rf"].sort_values("panel_rank")
	de = pd.read_csv(out_dir / "alzheimers_de_full.csv").set_index("gene")

	only_rec = rec.loc[~rec["in_other_panel"], "gene"].tolist()
	both = rec.loc[rec["in_other_panel"], "gene"].tolist()
	only_rf = rf.loc[~rf["in_other_panel"], "gene"].tolist()

	fig = plt.figure(figsize=(12.0, 13.0), dpi=300)
	ax = fig.add_axes([0.0, 0.42, 1.0, 0.58])
	r = 1.0
	cx_rec, cx_rf = -0.55, 0.55
	for cx, color in ((cx_rec, RECURSIEVE), (cx_rf, PLAIN_RF)):
		ax.add_patch(Circle((cx, 0), r, facecolor=color, alpha=0.22, lw=0))
		ax.add_patch(Circle((cx, 0), r, facecolor="none", edgecolor=color, lw=4.0))

	count = dict(ha="center", va="center", fontsize=34, fontweight="bold", color=INK)

	def card(x, y, text, edge):
		ax.text(x, y, text, ha="center", va="top", fontsize=14, color=INK,
				linespacing=1.35, bbox=dict(boxstyle="round,pad=0.5", fc="white",
											ec=edge, lw=2.0))

	# Same DE status split as alzheimers_venn.png (p takes precedence)
	criteria = (
		("DE", "p < 0.05\n|log$_2$FC| ≥ 0.1", 3),
		("fails_logfc", "p < 0.05\n|log$_2$FC| < 0.1", 3),
		("fails_pval", "p ≥ 0.05", 2),
	)
	for x, genes, edge in ((cx_rec - 0.45, only_rec, RECURSIEVE), (0, both, INK_2),
						   (cx_rf + 0.45, only_rf, PLAIN_RF)):
		d = de.loc[genes]
		status = pd.Series("DE", index=d.index).where(
			d["is_de"], d["pval"].lt(PVAL_CUTOFF).map({True: "fails_logfc", False: "fails_pval"}))
		ax.text(x, 0.55, f"{len(genes)}", **count)
		y = 0.36
		for key, label, n_lines in criteria:
			n = int((status == key).sum())
			if n == 0:
				continue
			card(x, y, f"$\\mathbf{{{n}}}$\n{label}", edge)
			y -= 0.1 * n_lines + 0.05

	for x, title, color in (
		(cx_rec - 0.25, "RecurSieve", RECURSIEVE),
		(cx_rf + 0.25, f"Plain RF top {len(rf)}", PLAIN_RF),
	):
		ax.text(x, r + 0.12, title, ha="center", va="bottom", fontsize=22,
				fontweight="bold", color=color)
	ax.set_xlim(-1.75, 1.75)
	ax.set_ylim(-1.1, 1.4)
	ax.set_aspect("equal")
	ax.axis("off")

	# Gene lists beneath each region
	ax_l = fig.add_axes([0.04, 0.02, 0.92, 0.39])
	ax_l.axis("off")
	cols = (
		(0.12, "RecurSieve", RECURSIEVE, only_rec),
		(0.50, "Shared", INK, both),
		(0.88, "Plain RF only", PLAIN_RF, only_rf),
	)
	n_rows = max(len(c[3]) for c in cols)
	step = 0.88 / n_rows
	for x, header, color, genes in cols:
		ax_l.text(x, 1.0, header, ha="center", va="top", fontsize=16,
				  fontweight="bold", color=color)
		for i, g in enumerate(genes):
			ax_l.text(x, 0.93 - i * step, g, ha="center",
					  va="top", fontsize=12.5, color=INK)

	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"alzheimers_rf_vs_recursieve_venn.{ext}",
					dpi=300, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	main()
