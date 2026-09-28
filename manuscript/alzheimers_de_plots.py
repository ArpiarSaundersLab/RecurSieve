"""Volcano plot and Venn diagram comparing Wilcoxon DE vs recursieve genes.

Recomputes only the Wilcoxon DE step (not the recursieve RF recursion) using
recursieve's own preprocessing/filtering/DE methods on the same 50,000-cell
subsample as alzheimers_analysis.py, then combines it with the saved
recursieve gene list (alzheimers_genes.csv).

Outputs (written next to this script):
- alzheimers_de_full.csv: Full Wilcoxon table (all tested genes) with DE flag
- alzheimers_recursieve_in_de.csv: Each recursieve gene with its DE stats
- alzheimers_volcano.png / .pdf
- alzheimers_venn.png / .pdf
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from recursieve import recursieve

# Palette (validated reference palette: diverging blue<->red, neutral gray)
UP = "#e34948"
DOWN = "#2a78d6"
NS = "#c9c8c3"
INK = "#0b0b0b"
INK_2 = "#52514e"
# Venn set colours: deliberately not red/blue, which mean up/down in the volcano
DE_SET = "#6b6a66"
RECURSIEVE = "#4a3aa7"

PVAL_CUTOFF = 0.05
REC_S = 520  # recursieve marker size in panel B
KEY_S = 300  # marker size in the gene key
LOGFC_CUTOFF = 0.1


def de_status(df: pd.DataFrame) -> pd.Series:
	"""Label each gene as DE or by which DE cutoff it fails (p takes precedence)."""
	status = np.where(df["is_de"], "DE",
					  np.where(df["pval"] < PVAL_CUTOFF, "fails_logfc", "fails_pval"))
	return pd.Series(status, index=df.index)


def compute_full_de(adata_path: Path) -> pd.DataFrame:
	"""Run recursieve's preprocessing and Wilcoxon DE without the RF recursion."""
	adata = ad.read_h5ad(adata_path)
	adata = sc.pp.subsample(adata, n_obs=50000, copy=True)
	adata.var_names = adata.var["feature_name"]
	adata.var_names_make_unique()

	# Bypass __init__ so ensemble_recursion() never runs
	model = recursieve.__new__(recursieve)
	model.adata = adata
	model.annsql_db = None
	model.group1, model.group2 = "low", "high"
	model.field_name = "Disease.Group"
	model.pval_cutoff, model.logfc_cutoff = PVAL_CUTOFF, LOGFC_CUTOFF
	model.preprocessing()
	model.filter_groups()
	model.scanpy_de_original_groups()

	df = sc.get.rank_genes_groups_df(model.adata, group=model.group2)
	df = df.rename(columns={
		"names": "gene", "logfoldchanges": "logfc", "pvals": "pval",
		"pvals_adj": "pval_adj", "scores": "score",
	})
	df["is_de"] = df["gene"].isin(model.de_dict.keys())
	return df


def check_reproduction(de_full: pd.DataFrame, saved_path: Path) -> None:
	"""Confirm recomputed stats match the DE rows saved by the original run."""
	saved = pd.read_csv(saved_path)
	merged = saved.merge(de_full, on="gene", suffixes=("_saved", ""))
	assert len(merged) == len(saved), "Saved DE genes missing from recomputed DE"
	assert np.allclose(merged["logfc_saved"], merged["logfc"], rtol=1e-4)
	assert np.allclose(merged["pval_saved"], merged["pval"], rtol=1e-4)
	print(f"Reproduction check passed ({len(saved)} saved DE rows match).")


def _separate_circles(pts, r, n_iter=500):
	"""Push overlapping circles (display px) apart just enough to not overlap."""
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


def _panel_title(ax, letter, title):
	"""Centered bold title with the panel letter at the top-left corner."""
	ax.set_title(title, fontsize=20, fontweight="bold", color=INK, pad=14)
	ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=26, fontweight="bold",
			color=INK, ha="right", va="bottom")


def _style_axes(ax):
	"""Recessive spines and ticks."""
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK)
		ax.spines[side].set_linewidth(4.0)
	ax.tick_params(colors=INK, width=4.0, length=11, labelsize=21)


def _draw_volcano_layer(ax, df, xlim, ylim, pin_offscale=True, ymin=0.0, size_scale=1.0,
						fc_lines=True):
	"""Scatter genes within xlim; points above ylim are pinned to the top as triangles
	(or dropped if pin_offscale is False)."""
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
	# \\mathbf keeps the mathtext sub/superscripts bold like the rest of the label
	ax.set_xlabel(r"log$_\mathbf{2}$ fold change (high vs low)")
	ax.set_ylabel(r"$\mathbf{-}$log$_\mathbf{10}$ p-value (Wilcoxon)")
	_style_axes(ax)
	return x, y


def plot_volcano(de_full: pd.DataFrame, rec_genes: list[str], out_stem: Path) -> None:
	"""Two-panel volcano: (A) all tested genes, (B) zoom on recursieve genes."""
	df = de_full.copy()
	# p-values can underflow to 0; floor at smallest positive value
	floor = df.loc[df["pval"] > 0, "pval"].min()
	df["nlp"] = -np.log10(df["pval"].clip(lower=floor))
	is_rec = df["gene"].isin(rec_genes)

	fig, (ax_a, ax_b, ax_key) = plt.subplots(
		1, 3, figsize=(18.5, 9.8), dpi=300,
		gridspec_kw={"width_ratios": [1, 1.5, 0.62]})
	# fix geometry up front: circle de-overlap runs in display space, so the axes
	# must not be resized afterwards (no tight_layout)
	fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.34, wspace=0.18)

	# Panel A: full volcano
	# cap y just above the runner-up so the single extreme gene doesn't squash the rest
	y_cap = 120
	xa, ya = _draw_volcano_layer(ax_a, df, (-2.5, 2.5), (0, y_cap), size_scale=1.8,
								 fc_lines=False)
	for _, r in df[df["nlp"] > y_cap].iterrows():
		ax_a.annotate(f"↑{r['nlp']:.0f}", (xa[r.name], ya[r.name]),
					  xytext=(9, 0), textcoords="offset points", fontsize=11,
					  color=INK, va="center")
	ax_a.scatter(xa[is_rec], ya[is_rec], s=55, facecolors="none", edgecolors=INK,
				 linewidths=1.2, zorder=5, label="recursieve gene")
	zx, zy = (-0.35, 0.75), (0, 80)
	ax_a.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
								 fill=False, ec=INK_2, lw=1.2, ls=":", zorder=6))
	ax_a.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.2),
				ncol=2, fontsize=17, markerscale=2.2, columnspacing=1.2,
				handletextpad=0.3, labelspacing=0.6)
	_panel_title(ax_a, "A", "All Tested Genes (Top 2,000 HVGs)")

	# Panel B: zoom on recursieve genes
	xb, yb = _draw_volcano_layer(ax_b, df, zx, zy, pin_offscale=False, ymin=-0.3,
								 size_scale=4.0)
	# sqrt y-scale spreads the crowded low-p region where most recursieve genes sit
	ssqrt = (lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda v: np.sign(v) * v ** 2)
	ax_b.set_yscale("function", functions=ssqrt)
	ax_b.set_yticks([0, 2, 5, 10, 20, 40, 60, 80])
	ax_b.set_ylim(top=95)  # headroom for off-scale labels
	ax_b.set_ylabel("")  # same quantity as A; the sqrt scale is noted under the panel
	# number genes by recursieve selection order (1 = first selected)
	number = {g: k + 1 for k, g in enumerate(rec_genes)}
	rec = df[is_rec].copy()
	rec["num"] = rec["gene"].map(number)
	rec = rec.sort_values("num")
	off = (rec["nlp"] > zy[1]).to_numpy()
	rx, ry = xb[rec.index].to_numpy(), yb[rec.index].to_numpy()
	fill = np.where(~rec["is_de"], NS, np.where(rec["logfc"] > 0, UP, DOWN))
	ink = np.where(fill == NS, INK, "white")

	# nudge overlapping circles apart; a leader line marks the true position
	r_px = np.sqrt(REC_S) / 2 * fig.dpi / 72
	true_px = ax_b.transData.transform(np.column_stack([rx, ry]))
	# allow ~12% overlap between neighbours: keeps numbers legible while holding
	# circles close to their true positions
	pos_px = _separate_circles(true_px, r_px * 0.88)
	cx, cy = ax_b.transData.inverted().transform(pos_px).T
	for k in np.flatnonzero(np.hypot(*(pos_px - true_px).T) > 1):
		ax_b.plot([rx[k], cx[k]], [ry[k], cy[k]], color=INK, lw=0.9, zorder=4)
		ax_b.scatter(rx[k], ry[k], s=14, c=INK, linewidths=0, zorder=4)
	ax_b.scatter(cx, cy, s=REC_S, c=fill, edgecolors=INK, linewidths=1.4, zorder=5)
	for x, y, n, c in zip(cx, cy, rec["num"], ink):
		ax_b.text(x, y, str(n), ha="center", va="center", fontsize=13.5,
				  fontweight="bold", color=c, zorder=6)
	# off-scale values above the circles, alternating sides so neighbours don't clash
	order = np.argsort(cx[off])
	for side, (x, y, nlp) in zip(np.resize([-1, 1], off.sum()),
								 np.array(list(zip(cx[off], cy[off], rec.loc[off, "nlp"])))[order]):
		ax_b.annotate(f"↑{nlp:.0f}", (x, y), xytext=(side * 4, 10),
					  textcoords="offset points", ha="right" if side < 0 else "left",
					  va="bottom", fontsize=10.5, color=INK)

	# key: number -> gene
	ax_key.axis("off")
	ax_key.set_xlim(0, 1)
	per_col = int(np.ceil(len(rec) / 2))
	ax_key.set_ylim(per_col + 0.5, -1.6)
	ax_key.text(0.0, -1.1, "recursieve genes", fontsize=13, fontweight="bold",
				color=INK, va="center")
	ax_key.text(0.0, -0.3, "# = selection order", fontsize=11, color=INK_2, va="center")
	for k, g in enumerate(rec.itertuples()):
		x0, row = (k // per_col) * 0.5, k % per_col + 1
		ax_key.scatter(x0 + 0.05, row, s=KEY_S, c=fill[k], edgecolors=INK,
					   linewidths=1.0, clip_on=False)
		ax_key.text(x0 + 0.05, row, str(g.num), ha="center", va="center", fontsize=10,
					fontweight="bold", color=ink[k])
		ax_key.text(x0 + 0.12, row, g.gene, fontsize=12.5, color=INK, va="center")

	notes = ["Square-root y-axis", "Circle fill = DE status (as in A)",
			 "↑n: off-scale, true $-$log$_{10}$p = n"]
	ax_b.legend([Line2D([], [], ls="none")] * len(notes), notes, frameon=False,
				loc="upper left", bbox_to_anchor=(0.03, -0.2), ncol=1, fontsize=17,
				handlelength=0, handletextpad=0, labelspacing=0.6)
	_panel_title(ax_b, "B", "Recursieve Genes")

	# bold tick values; done last because set_yscale/set_yticks rebuild the labels
	for ax in (ax_a, ax_b):
		plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontweight="bold")

	for ext in ("png", "pdf"):
		fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_venn(de_genes: set[str], rec_genes: set[str], n_fail_fc: int,
			  n_fail_p: int, out_stem: Path) -> None:
	"""Two-set Venn: DE genes vs recursieve genes, styled to match the volcano."""
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
	# region notes as small boxed cards: bold count (if any) over the criteria
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
	"""Recompute DE, write gene tables, and render volcano and Venn plots."""
	plt.rcParams.update({
		"font.family": "Arial", "font.size": 13, "axes.labelsize": 22,
		"axes.labelweight": "bold",
		# mathtext (log$_2$ etc.) in Arial too, and editable TrueType text in PDFs
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	out_dir = Path(__file__).resolve().parent
	data_dir = out_dir.parent / "data"
	de_full_path = out_dir / "alzheimers_de_full.csv"

	if de_full_path.exists():
		print(f"Using cached DE table: {de_full_path}")
		de_full = pd.read_csv(de_full_path)
	else:
		de_full = compute_full_de(data_dir / "alz.h5ad")
		de_full.to_csv(de_full_path, index=False)
		print(f"Saved full DE table: {de_full_path}")
	check_reproduction(de_full, out_dir / "alzheimers_de_results.csv")

	genes_df = pd.read_csv(out_dir / "alzheimers_genes.csv")
	rec_genes = list(dict.fromkeys(genes_df["gene"]))
	unique_panel = set(pd.read_csv(out_dir / "alzheimers_unique_genes.csv")["gene"])

	# recursieve genes with their DE stats (NaN if gene was not tested)
	rec_table = (pd.DataFrame({"gene": rec_genes, "rf_rank": range(len(rec_genes))})
				 .merge(de_full[["gene", "logfc", "pval", "pval_adj", "score", "is_de"]],
						on="gene", how="left"))
	rec_table["tested_in_de"] = rec_table["pval"].notna()
	rec_table["is_de"] = rec_table["is_de"].fillna(False).astype(bool)
	rec_table["de_status"] = de_status(rec_table)
	rec_table["in_unique_panel"] = rec_table["gene"].isin(unique_panel)
	rec_path = out_dir / "alzheimers_recursieve_in_de.csv"
	rec_table.to_csv(rec_path, index=False)
	print(f"Saved recursieve-in-DE table: {rec_path}")

	plot_volcano(de_full, rec_genes, out_dir / "alzheimers_volcano")
	de_set = set(de_full.loc[de_full["is_de"], "gene"])
	counts = rec_table["de_status"].value_counts()
	plot_venn(de_set, set(rec_genes), int(counts.get("fails_logfc", 0)),
			  int(counts.get("fails_pval", 0)), out_dir / "alzheimers_venn")
	print("Saved alzheimers_volcano.{png,pdf} and alzheimers_venn.{png,pdf}")


if __name__ == "__main__":
	main()
