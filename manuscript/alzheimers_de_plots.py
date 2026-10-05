"""Volcano plot and Venn diagram of Wilcoxon DE vs recursieve genes.

Reads alzheimers_de_full.csv, alzheimers_genes.csv, and
alzheimers_unique_genes.csv written by alzheimers_analysis.py.

Usage:
	python alzheimers_de_plots.py [--out-dir DIR]

Outputs (to DIR):
- alzheimers_recursieve_in_de.csv: Each recursieve gene with its DE stats
- alzheimers_volcano.png / .pdf
- alzheimers_venn.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

UP = "#e34948"
DOWN = "#2a78d6"
NS = "#c9c8c3"
INK = "#0b0b0b"
INK_2 = "#52514e"
# not red or blue, which mean up and down in the volcano
DE_SET = "#6b6a66"
RECURSIEVE = "#4a3aa7"
# recursieve genes that are not DE
LOW_FC = "#36d636"

PVAL_CUTOFF = 0.05
REC_S = 420  # gene marker size
KEY_S = 260  # key marker size
LOGFC_CUTOFF = 0.1


def de_status(df: pd.DataFrame) -> pd.Series:
	"""DE, fails_logfc, or fails_pval (p is checked first)."""
	status = np.where(df["is_de"], "DE",
					  np.where(df["pval"] < PVAL_CUTOFF, "fails_logfc", "fails_pval"))
	return pd.Series(status, index=df.index)


def _separate_circles(pts, r, n_iter=500):
	"""Push overlapping circles apart (display pixels)."""
	pos = pts.astype(float).copy()
	for _ in range(n_iter):
		moved = False
		for i in range(len(pos)):
			for j in range(i + 1, len(pos)):
				d = pos[i] - pos[j]
				dist = np.hypot(*d)
				if dist < 2 * r:
					u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
					step = (2 * r - dist) / 2 + 0.1
					pos[i] += u * step
					pos[j] -= u * step
					moved = True
		if not moved:
			break
	return pos


def _panel_title(ax, title, subtitle=None, fontsize=15):
	"""Bold title with an optional subtitle below it."""
	if subtitle is None:
		ax.set_title(title, fontsize=fontsize, fontweight="bold", color=INK, pad=10)
		return
	pad = 8
	ax.set_title(subtitle, fontsize=fontsize - 1, color=INK, pad=pad, linespacing=1.2)
	n_lines = subtitle.count("\n") + 1
	ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction",
				xytext=(0, pad + n_lines * (fontsize - 1) * 1.2 + 2), textcoords="offset points",
				ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=INK)


def _style_axes(ax):
	"""Axis style."""
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK)
		ax.spines[side].set_linewidth(1.3)
	ax.tick_params(colors=INK, width=1.3, length=5, labelsize=13)


def _draw_volcano_layer(ax, df, xlim, ylim, pin_offscale=True, ymin=0.0, size_scale=1.0,
						fc_lines=True):
	"""Scatter genes in xlim. Points above ylim are drawn as triangles at the top."""
	x = df["logfc"]
	y = df["nlp"].clip(upper=ylim[1])
	in_x = x.between(*xlim)
	off = df["nlp"] > ylim[1]
	up = df["is_de"] & (df["logfc"] > 0)
	down = df["is_de"] & (df["logfc"] < 0)
	ns = ~df["is_de"]
	for mask, color, size, lab in (
		(ns, NS, 6, f"Not significant ({ns.sum():,})"),
		(down, DOWN, 9, f"Down in high ({down.sum()})"),
		(up, UP, 9, f"Up in high ({up.sum()})"),
	):
		size = size * size_scale
		m = mask & in_x & ~off
		ax.scatter(x[m], y[m], s=size, c=color, linewidths=0, alpha=0.85,
				   rasterized=True, label=lab, zorder=2)
		m = mask & in_x & off
		if not pin_offscale:
			continue
		ax.scatter(x[m], y[m], s=size * 2.5, c=color, marker="^", linewidths=0,
				   alpha=0.85, rasterized=True, zorder=2)
	ax.axhline(-np.log10(PVAL_CUTOFF), color=INK_2, lw=1.0, ls="--", zorder=1)
	for v in (-LOGFC_CUTOFF, LOGFC_CUTOFF) if fc_lines else ():
		ax.axvline(v, color=INK_2, lw=1.0, ls="--", zorder=1)
	ax.set_xlim(xlim[0] - 0.03 * (xlim[1] - xlim[0]), xlim[1] + 0.03 * (xlim[1] - xlim[0]))
	ax.set_ylim(ymin, ylim[1] * 1.04)
	ax.set_xlabel(r"log$_\mathbf{2}$ fold change (high vs low)")
	ax.set_ylabel(r"$\mathbf{-}$log$_\mathbf{10}$ p-value (Wilcoxon)")
	_style_axes(ax)
	return x, y


def plot_volcano(de_full: pd.DataFrame, rec_genes: list[str], out_stem: Path) -> None:
	"""Volcano of all genes and a zoom on recursieve genes."""
	df = de_full.copy()
	# p-values of 0 are set to the smallest positive p
	floor = df.loc[df["pval"] > 0, "pval"].min()
	df["nlp"] = -np.log10(df["pval"].clip(lower=floor))
	is_rec = df["gene"].isin(rec_genes)

	fig, (ax_a, ax_b, ax_key) = plt.subplots(
		1, 3, figsize=(18.5, 9.8), dpi=300,
		gridspec_kw={"width_ratios": [1, 1.5, 0.62]})
	# fixed layout: circle spacing is computed in display space
	fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.34, wspace=0.18)

	# all genes, y capped so one extreme gene does not compress the rest
	y_cap = 120
	xa, ya = _draw_volcano_layer(ax_a, df, (-2.5, 2.5), (0, y_cap), size_scale=1.8,
								 fc_lines=False)
	for _, r in df[df["nlp"] > y_cap].iterrows():
		ax_a.annotate(f"↑{r['nlp']:.0f}", (xa[r.name], ya[r.name]),
					  xytext=(9, 0), textcoords="offset points", fontsize=11,
					  color=INK, va="center")
	ax_a.scatter(xa[is_rec], ya[is_rec], s=55, facecolors="none", edgecolors=INK,
				 linewidths=1.2, zorder=5)
	zx, zy = (-0.35, 0.75), (0, 80)
	ax_a.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
								 fill=False, ec=INK_2, lw=1.2, ls=":", zorder=6))
	_panel_title(ax_a, "Alzheimer's Cohort",
				 "Differential Expression (Wilcoxon)\nLow vs High Pathology")

	# zoom on recursieve genes
	xb, yb = _draw_volcano_layer(ax_b, df, zx, zy, pin_offscale=False, ymin=-0.3,
								 size_scale=4.0)
	# sqrt y scale spreads out the low p region
	ssqrt = (lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda v: np.sign(v) * v ** 2)
	ax_b.set_yscale("function", functions=ssqrt)
	ax_b.set_yticks([0, 2, 5, 10, 20, 40, 60, 80])
	ax_b.set_ylim(top=95)
	ax_b.set_ylabel("")
	# numbers follow selection order
	number = {g: k + 1 for k, g in enumerate(rec_genes)}
	rec = df[is_rec].copy()
	rec["num"] = rec["gene"].map(number)
	rec = rec.sort_values("num")
	off = (rec["nlp"] > zy[1]).to_numpy()
	rx, ry = xb[rec.index].to_numpy(), yb[rec.index].to_numpy()
	fill = np.where(~rec["is_de"], LOW_FC, np.where(rec["logfc"] > 0, UP, DOWN))
	ink = np.where(fill == LOW_FC, INK, "white")

	# separate overlapping circles; a line marks the true position
	r_px = np.sqrt(REC_S) / 2 * fig.dpi / 72
	true_px = ax_b.transData.transform(np.column_stack([rx, ry]))
	# allow about 12% overlap
	pos_px = _separate_circles(true_px, r_px * 0.88)
	cx, cy = ax_b.transData.inverted().transform(pos_px).T
	for k in np.flatnonzero(np.hypot(*(pos_px - true_px).T) > 1):
		ax_b.plot([rx[k], cx[k]], [ry[k], cy[k]], color=INK, lw=0.9, zorder=4)
		ax_b.scatter(rx[k], ry[k], s=14, c=INK, linewidths=0, zorder=4)
	ax_b.scatter(cx, cy, s=REC_S, c=fill, edgecolors=INK, linewidths=1.4, zorder=5)
	for x, y, n, c in zip(cx, cy, rec["num"], ink):
		ax_b.text(x, y, str(n), ha="center", va="center", fontsize=11.5,
				  fontweight="bold", color=c, zorder=6)
	# off-scale values, alternating sides
	order = np.argsort(cx[off])
	for side, (x, y, nlp) in zip(np.resize([-1, 1], off.sum()),
								 np.array(list(zip(cx[off], cy[off], rec.loc[off, "nlp"])))[order]):
		ax_b.annotate(f"↑{nlp:.0f}", (x, y), xytext=(side * 4, 10),
					  textcoords="offset points", ha="right" if side < 0 else "left",
					  va="bottom", fontsize=11, color=INK)

	# gene key
	ax_key.axis("off")
	ax_key.set_xlim(0, 1)
	per_col = int(np.ceil(len(rec) / 2))
	ax_key.set_ylim(per_col + 0.5, 0.4)
	for k, g in enumerate(rec.itertuples()):
		x0, row = (k // per_col) * 0.5, k % per_col + 1
		ax_key.scatter(x0 + 0.05, row, s=KEY_S, c=fill[k], edgecolors=INK,
					   linewidths=1.0, clip_on=False)
		ax_key.text(x0 + 0.05, row, str(g.num), ha="center", va="center", fontsize=9.5,
					fontweight="bold", color=ink[k])
		ax_key.text(x0 + 0.12, row, g.gene, fontsize=13, color=INK, va="center")

	_panel_title(ax_b, "RecurSieve Genes")

	# shared legend
	def dot(color, edge="none"):
		return Line2D([], [], ls="none", marker="o", markersize=13, markerfacecolor=color,
					  markeredgecolor=edge, markeredgewidth=1.2)
	n_up = int((df["is_de"] & (df["logfc"] > 0)).sum())
	n_down = int((df["is_de"] & (df["logfc"] < 0)).sum())
	n_low = int((~rec["is_de"]).sum())
	handles = [dot(NS), dot(DOWN), dot(LOW_FC, INK), dot(UP), dot("none", INK),
			   Line2D([], [], ls="none")]
	labels = [f"Not significant (NS) ({int((~df['is_de']).sum()):,})",
			  f"Down in high ({n_down})",
			  f"RecurSieve ({n_low})\n$\\it{{Low\\ LogFC}}$",
			  f"Up in high ({n_up})", "RecurSieve gene", "↑n: off-scale"]
	fig.legend(handles, labels, frameon=False, loc="upper left",
			   bbox_to_anchor=(0.1, 0.22), ncol=2, fontsize=15, columnspacing=3.0,
			   handletextpad=0.6, labelspacing=1.0)

	for ext in ("png", "pdf"):
		fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_venn(de_genes: set[str], rec_genes: set[str], n_fail_fc: int,
			  n_fail_p: int, out_stem: Path) -> None:
	"""Venn of DE genes vs recursieve genes."""
	both = de_genes & rec_genes
	only_de = de_genes - rec_genes
	only_rec = rec_genes - de_genes

	fig, ax = plt.subplots(figsize=(10.0, 8.0), dpi=300)
	r = 1.0
	cx_de, cx_rec = -0.55, 0.55
	for cx, color in ((cx_de, DE_SET), (cx_rec, RECURSIEVE)):
		ax.add_patch(Circle((cx, 0), r, facecolor=color, alpha=0.22, lw=0))
		ax.add_patch(Circle((cx, 0), r, facecolor="none", edgecolor=color, lw=4.0))

	count = dict(ha="center", va="center", fontsize=34, fontweight="bold", color=INK)
	ax.text(cx_de - 0.45, 0.36, f"{len(only_de)}", **count)
	ax.text(0, 0, f"{len(both)}", **count)
	ax.text(cx_rec + 0.42, 0.36, f"{len(only_rec)}", **count)
	# boxed cards with each region's criteria
	def card(x, y, text, edge):
		ax.text(x, y, text, ha="center", va="top", fontsize=17, color=INK,
				linespacing=1.35, bbox=dict(boxstyle="round,pad=0.55", fc="white",
											ec=edge, lw=2.0))

	card(cx_de - 0.45, 0.12, "p < 0.05\n|log$_2$FC| ≥ 0.1", DE_SET)
	card(cx_rec + 0.42, 0.12,
		 f"$\\mathbf{{{n_fail_fc}}}$\np < 0.05\n|log$_2$FC| < 0.1", RECURSIEVE)
	card(cx_rec + 0.42, -0.38, f"$\\mathbf{{{n_fail_p}}}$\np ≥ 0.05", RECURSIEVE)

	for x, title, color in (
		(cx_de - 0.25, f"Wilcoxon DE (n = {len(de_genes)})", DE_SET),
		(cx_rec + 0.25, f"Recursieve (n = {len(rec_genes)})", RECURSIEVE),
	):
		ax.text(x, r + 0.12, title, ha="center", va="bottom", fontsize=22,
				fontweight="bold", color=color)

	ax.set_xlim(-1.75, 1.75)
	ax.set_ylim(-1.1, 1.4)
	ax.set_aspect("equal")
	ax.axis("off")
	fig.tight_layout()
	for ext in ("png", "pdf"):
		fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	plt.rcParams.update({
		"font.family": "Arial", "font.size": 13, "axes.labelsize": 16,
		"axes.labelweight": "bold",
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-dir", type=Path,
						default=Path(__file__).resolve().parent / "robustness" / "all_cell_types")
	out_dir = parser.parse_args().out_dir
	de_full = pd.read_csv(out_dir / "alzheimers_de_full.csv")

	genes_df = pd.read_csv(out_dir / "alzheimers_genes.csv")
	rec_genes = list(dict.fromkeys(genes_df["gene"]))
	unique_panel = set(pd.read_csv(out_dir / "alzheimers_unique_genes.csv")["gene"])

	# NaN if the gene was not tested
	rec_table = (pd.DataFrame({"gene": rec_genes, "rf_rank": range(len(rec_genes))})
				 .merge(de_full[["gene", "logfc", "pval", "pval_adj", "score", "is_de"]],
						on="gene", how="left"))
	rec_table["tested_in_de"] = rec_table["pval"].notna()
	rec_table["is_de"] = rec_table["is_de"].fillna(False).astype(bool)
	rec_table["de_status"] = de_status(rec_table)
	rec_table["in_unique_panel"] = rec_table["gene"].isin(unique_panel)
	rec_table.to_csv(out_dir / "alzheimers_recursieve_in_de.csv", index=False)

	plot_volcano(de_full, rec_genes, out_dir / "alzheimers_volcano")
	de_set = set(de_full.loc[de_full["is_de"], "gene"])
	counts = rec_table["de_status"].value_counts()
	plot_venn(de_set, set(rec_genes), int(counts.get("fails_logfc", 0)),
			  int(counts.get("fails_pval", 0)), out_dir / "alzheimers_venn")


if __name__ == "__main__":
	main()
