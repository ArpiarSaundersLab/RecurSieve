"""Staged gene recovery benchmark on simulated data.

Counts how many of the n_staged staged genes appear in each method's top N
panel. Staged genes keep each cell's library size.

Scenarios:
  null          labels shuffled, no staged genes
  bimodal       group_2 has two modes around the group_1 mean
  trimodal      group_2 has three modes around the group_1 mean
  skew          group_1 right-tailed, group_2 left-tailed, same mean
  coexpression  same marginals, staged genes correlated only in group_2

Trial i uses sim_seed + i.

Outputs (to --out-dir):
  feature_comparison_wide_counts.csv
  feature_comparison_wide_heatmap.png / .pdf

Usage:
  python feature_comparision.py --seeds 5
  python feature_comparision.py --scenarios bimodal --seeds 3
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import skewnorm

from scsim import scsim
import recursieve

METHOD_ORDER = ["Wilcoxon", "TTest", "LogReg", "recursieve"]
SCENARIO_ORDER = ["null", "bimodal", "trimodal", "skew", "coexpression"]


def _background(sim_seed, ncells, ngenes):
	"""Simulated background counts."""
	sim = scsim(
		ngenes=ngenes, ncells=ncells, ngroups=2,
		libloc=7.64, libscale=0.78,
		mean_rate=7.68, mean_shape=0.34,
		expoutprob=0.00286, expoutloc=6.15, expoutscale=0.49,
		diffexpprob=0.025, diffexpdownprob=0., diffexploc=1.0, diffexpscale=1.0,
		bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=0,
		nproggenes=400, progdownprob=0., progdeloc=1.0, progdescale=1.0,
		progcellfrac=0.35, proggoups=None,
		minprogusage=0.1, maxprogusage=0.7, seed=sim_seed,
	)
	sim.simulate()
	a = ad.AnnData(X=sim.counts, obs=sim.cellparams, var=sim.geneparams)
	a.var_names = a.var_names.astype(str)
	a.obs["group"] = a.obs["group"].apply(
		lambda x: "group_1" if x == 1 else "group_2")
	return a


def _match_block_totals(pattern, totals, rng=None):
	"""Resample a staged pattern so each cell keeps its staged block total."""
	pattern = np.maximum(np.asarray(pattern, dtype=float), 0.0)
	s = pattern.sum(axis=1, keepdims=True)
	s[s == 0] = 1.0
	prob = pattern / s
	totals = np.asarray(totals, dtype=np.int64)
	if rng is None:
		rng = np.random.default_rng(0)
	out = np.empty(pattern.shape, dtype=np.int64)
	for i in range(pattern.shape[0]):
		out[i] = rng.multinomial(totals[i], prob[i])
	return out


def _staged_totals(adata, n_staged):
	"""Per-cell sum of the staged genes."""
	return np.asarray(adata.X)[:, :n_staged].sum(axis=1).astype(np.int64)


def build_null(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
				ngenes=2000, **_):
	"""Labels shuffled, no staged genes."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	adata.obs["group"] = pd.Categorical(
		rng.permutation(adata.obs["group"].values))
	return adata, []


def build_bimodal(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
				  ngenes=2000, center=8.0, sd=1.5, sep=4.0, **_):
	"""group_1 normal, group_2 bimodal about the same center."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())
	T = _staged_totals(adata, n_staged)
	p1 = rng.normal(center, sd, size=(n1, n_staged))
	hi = rng.random((n2, n_staged)) < 0.5
	p2 = rng.normal(np.where(hi, center + sep, center - sep), sd)
	g1 = _match_block_totals(p1, T[m1], rng)
	g2 = _match_block_totals(p2, T[~m1], rng)

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :n_staged], X[~m1, :n_staged] = g1, g2
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")
	return adata, list(adata.var_names[:n_staged])


def build_trimodal(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
					ngenes=2000, center=8.0, sd=1.5, sep=4.0, **_):
	"""group_1 normal, group_2 trimodal about the same center."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())
	T = _staged_totals(adata, n_staged)
	p1 = rng.normal(center, sd, size=(n1, n_staged))
	state = rng.integers(0, 3, size=(n2, n_staged))
	mu2 = np.select([state == 0, state == 1],
					[center - sep, center], default=center + sep)
	p2 = rng.normal(mu2, sd)
	g1 = _match_block_totals(p1, T[m1], rng)
	g2 = _match_block_totals(p2, T[~m1], rng)

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :n_staged], X[~m1, :n_staged] = g1, g2
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")
	return adata, list(adata.var_names[:n_staged])


def build_coexpression(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
					   ngenes=2000, **_):
	"""Same marginals in both groups, staged genes correlated only in group_2."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())
	lat1 = rng.negative_binomial(5, 0.5, size=(n1, n_staged)).astype(float)
	lat2 = rng.negative_binomial(5, 0.5, size=(n2, n_staged)).astype(float)
	# one shared cell order makes the group_2 genes rise together
	order = np.argsort(rng.standard_normal(n2))
	for j in range(n_staged):
		lat2[:, j] = np.sort(lat2[:, j])[order]
	T = _staged_totals(adata, n_staged)
	g1 = _match_block_totals(lat1, T[m1], rng)
	g2 = _match_block_totals(lat2, T[~m1], rng)

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :n_staged] = np.asarray(g1).astype(np.int64)
	X[~m1, :n_staged] = np.asarray(g2).astype(np.int64)
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")
	return adata, list(adata.var_names[:n_staged])


def build_skew(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
				ngenes=2000, center=8.0, sd=2.0, alpha=6.0, **_):
	"""group_1 right-tailed, group_2 left-tailed, same mean."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())
	T = _staged_totals(adata, n_staged)

	d = alpha / np.sqrt(1.0 + alpha ** 2)
	shift = sd * d * np.sqrt(2.0 / np.pi)

	def draw(a_shape, shape):
		loc = center - np.sign(a_shape) * shift
		return skewnorm.rvs(a_shape, loc=loc, scale=sd, size=shape,
							random_state=rng)

	g1 = _match_block_totals(draw(+alpha, (n1, n_staged)), T[m1], rng)
	g2 = _match_block_totals(draw(-alpha, (n2, n_staged)), T[~m1], rng)

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :n_staged], X[~m1, :n_staged] = g1, g2
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")
	return adata, list(adata.var_names[:n_staged])


BUILDERS = {"null": build_null, "bimodal": build_bimodal,
			"trimodal": build_trimodal, "skew": build_skew,
			"coexpression": build_coexpression}


def _normalized(adata):
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	return a


def _filter_ranked_genes(adata, method, n_top=50, pval_cutoff=0.05):
	"""Top ranked genes with p < pval_cutoff (logreg has no p-values)."""
	a = _normalized(adata)
	sc.tl.rank_genes_groups(a, groupby="group", groups=["group_2"],
							reference="group_1", method=method, use_raw=False)
	rankings = a.uns["rank_genes_groups"]
	names = np.asarray(rankings["names"]["group_2"])
	pvals = None
	for key in ("pvals", "pvals_adj"):
		if key not in rankings:
			continue
		candidate = rankings[key]
		if isinstance(candidate, dict):
			pvals = candidate.get("group_2", None)
		else:
			if getattr(candidate, "dtype", None) is not None and hasattr(candidate.dtype, "names"):
				pvals = candidate["group_2"] if "group_2" in candidate.dtype.names else None
			else:
				pvals = candidate
		if pvals is not None:
			break
	if pvals is not None:
		pvals = np.asarray(pvals, dtype=float)
		mask = np.isfinite(pvals) & (pvals < pval_cutoff)
		filtered = names[mask].tolist()
		return filtered[:n_top]
	return names[:n_top].tolist()


def run_wilcoxon(adata, n_top=50, pval_cutoff=0.05):
	return _filter_ranked_genes(adata, "wilcoxon", n_top=n_top,
								 pval_cutoff=pval_cutoff)


def run_ttest(adata, n_top=50, pval_cutoff=0.05):
	return _filter_ranked_genes(adata, "t-test", n_top=n_top,
								 pval_cutoff=pval_cutoff)


def run_logreg(adata, n_top=50, pval_cutoff=0.05):
	return _filter_ranked_genes(adata, "logreg", n_top=n_top,
								 pval_cutoff=pval_cutoff)


def run_recursieve(adata, n_top=50, seed=42, max_iterations=50, pval_cutoff=0.05):
	model = recursieve.recursieve(
		adata.copy(), group1="group_1", group2="group_2", field_name="group",
		max_iterations=max_iterations, flip_rate_percentage=0.01, seed=seed,
		pval_cutoff=pval_cutoff, print_to_console=False, plots=False,
	)
	return list(model.genes)[:n_top]


def count_recovered(panel, staged):
	"""Number of staged genes in the panel."""
	return int(sum(1 for g in panel if g in set(staged)))


_RAMP = ["#f4f7fb", "#dce7f3", "#bcd2e8", "#94b6d8", "#6b97c4",
		 "#4a79ac", "#325d8e", "#1f4370", "#132c4d"]
_INK, _MUTED, _SURFACE = "#1a1d21", "#5c636b", "#ffffff"


TITLE = "Genes Recovered"
SUBTITLE = "(Simulated)"
METHOD_LABELS = {"recursieve": "RecurSieve"}

def _normalize_scenario_labels(df):
	"""Keep "null" as a scenario name instead of NA."""
	df = df.copy()
	df["scenario"] = df["scenario"].astype("string")
	df["scenario"] = df["scenario"].fillna("null")
	df["scenario"] = df["scenario"].replace({"nan": "null", "NaN": "null", "None": "null", "": "null"})
	df["scenario"] = df["scenario"].astype(str)
	return df


def _mixed_text(fig, parts, x, y, transform, vertical=False, **kw):
	"""Draw (text, style) pieces end to end as one centered label."""
	renderer = fig.canvas.get_renderer()
	artists = [fig.text(0, 0, t, rotation=90 if vertical else 0,
						ha="left", va="bottom", **{**kw, **st}) for t, st in parts]
	sizes = [a.get_window_extent(renderer) for a in artists]
	lengths = [b.height if vertical else b.width for b in sizes]
	ax_px = transform.transform((x, y))
	start = (ax_px[1] if vertical else ax_px[0]) - sum(lengths) / 2
	for a, b, n in zip(artists, sizes, lengths):
		px = (ax_px[0] - b.width / 2, start) if vertical else (start, ax_px[1])
		a.set_position(fig.transFigure.inverted().transform(px))
		start += n


def plot_heatmap(df, out_dir, n_staged, scenarios):
	"""Mean genes recovered per scenario and method."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap

	plt.rcParams.update({
		"font.family": "Arial",
		"mathtext.fontset": "custom", "mathtext.rm": "Arial",
		"mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"pdf.fonttype": 42, "ps.fonttype": 42,
	})
	df = _normalize_scenario_labels(df)
	cols = [m for m in METHOD_ORDER if m in set(df["method"])]
	ordered_scenarios = [s for s in SCENARIO_ORDER if s in scenarios]
	missing = [s for s in scenarios if s not in ordered_scenarios]
	ordered_scenarios.extend(missing)
	piv = (df.pivot_table(index="scenario", columns="method",
						  values="n_recovered")[cols].reindex(ordered_scenarios, fill_value=0.0))
	M = np.nan_to_num(piv.to_numpy(dtype=float), nan=0.0)

	cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
	fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * M.shape[1] + 2.4),
									 0.85 * M.shape[0] + 1.9))
	fig.subplots_adjust(left=0.22, right=0.86, top=0.88, bottom=0.2)
	fig.patch.set_facecolor(_SURFACE); ax.set_facecolor(_SURFACE)
	mesh = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=n_staged,
						 edgecolors=_SURFACE, linewidth=2)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = M[i, j]
			if np.isfinite(v):
				label = "0.0" if abs(v) < 1e-12 else f"{v:.1f}"
				ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
						fontsize=8,
						color=_SURFACE if v > n_staged * 0.55 else _INK)

	ax.set_xticks(np.arange(M.shape[1]) + 0.5)
	ax.set_xticklabels([METHOD_LABELS.get(c, c) for c in cols], rotation=35,
					   ha="right", fontsize=14, color=_INK)
	ax.set_yticks(np.arange(M.shape[0]) + 0.5)
	ax.set_yticklabels(list(piv.index), fontsize=13, color=_INK)
	ax.invert_yaxis()
	for sp in ax.spines.values():
		sp.set_visible(False)
	ax.tick_params(length=0)

	cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
	cbar.set_ticks(np.arange(0, n_staged + 1, 2))
	_mixed_text(fig, [("mean genes recovered ", {}), (f"(of {n_staged})", {"fontstyle": "italic"})],
				4.2, 0.5, cbar.ax.transAxes, vertical=True, fontsize=12, color=_INK)
	cbar.outline.set_visible(False)
	cbar.ax.tick_params(length=0, labelsize=15, colors=_INK)

	_mixed_text(fig, [(TITLE + " ", {"fontweight": "bold"}), (SUBTITLE, {"fontstyle": "italic"})],
				0.5, 1.05, ax.transAxes, fontsize=14, color=_INK)
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"feature_comparison_wide_heatmap.{ext}",
					dpi=300, facecolor=_SURFACE)
	plt.close(fig)


def ordered_scenarios(df, scenarios):
	"""Scenarios in SCENARIO_ORDER, then any others."""
	seen = list(dict.fromkeys(list(df["scenario"]) + scenarios))
	order = [x for x in SCENARIO_ORDER if x in seen]
	return order + [x for x in seen if x not in order and x in BUILDERS]


def main():
	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--seeds", type=int, default=5, help="independent trials")
	ap.add_argument("--n-staged", type=int, default=10)
	ap.add_argument("--n-top", type=int, default=50, help="panel size per method")
	ap.add_argument("--pval-cutoff", type=float, default=0.05,
					help="keep only genes with p < cutoff for DE methods")
	ap.add_argument("--ncells", type=int, default=2000)
	ap.add_argument("--ngenes", type=int, default=2000)
	ap.add_argument("--sim-seed", type=int, default=32,
					help="base seed; trial i uses sim_seed+i")
	ap.add_argument("--max-iterations", type=int, default=50)
	ap.add_argument("--center", type=float, default=8.0)
	ap.add_argument("--sd", type=float, default=1.5)
	ap.add_argument("--sep", type=float, default=4.0,
					help="bimodal: distance from center to each mode")
	ap.add_argument("--alpha", type=float, default=6.0,
					help="skew: skew-normal shape; group_1 +alpha, group_2 -alpha")
	ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "robustness" / "feature_comparison"))
	ap.add_argument("--plot-only", action="store_true",
					help="plot the saved CSV without rerunning")
	args = ap.parse_args()

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	bad = [x for x in scenarios if x not in BUILDERS]
	if bad:
		raise SystemExit(f"unknown scenarios {bad}; choose from {sorted(BUILDERS)}")

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	csv = out_dir / "feature_comparison_wide_counts.csv"
	columns = ["scenario", "seed", "method", "n_recovered", "panel_size"]
	df = pd.DataFrame(columns=columns)
	if csv.exists():
		df = _normalize_scenario_labels(pd.read_csv(csv, keep_default_na=False, na_values=[]))

	if not args.plot_only:
		rows = []
		for scen in scenarios:
			for seed in range(args.seeds):
				t0 = time.time()
				adata, staged = BUILDERS[scen](
					sim_seed=args.sim_seed + seed, stage_seed=seed,
					n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes,
					center=args.center, sd=args.sd, sep=args.sep, alpha=args.alpha)
				panels = {
					"Wilcoxon": run_wilcoxon(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
					"TTest": run_ttest(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
					"LogReg": run_logreg(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
					"recursieve": run_recursieve(adata, n_top=args.n_top,
												 max_iterations=args.max_iterations,
												 pval_cutoff=args.pval_cutoff),
				}
				line = []
				for m in METHOD_ORDER:
					n = count_recovered(panels[m], staged)
					rows.append(dict(scenario=scen, seed=seed, method=m,
									 n_recovered=n, panel_size=len(panels[m])))
					line.append(f"{m}={n}")
				print(f"  [{scen} s{seed}] " + "  ".join(line) + f"   [{time.time()-t0:.0f}s]", flush=True)
		df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
		df = df.drop_duplicates(subset=["scenario", "seed", "method"], keep="last").reset_index(drop=True)
		df.to_csv(csv, index=False)

	if df.empty:
		raise SystemExit(f"No benchmark data in {csv}")
	order = ordered_scenarios(df, scenarios)
	piv = df.pivot_table(index="scenario", columns="method", values="n_recovered")
	print(piv[[m for m in METHOD_ORDER if m in piv.columns]].reindex(order).round(2).to_string())
	plot_heatmap(df, out_dir, args.n_staged, order)

if __name__ == "__main__":
	main()
