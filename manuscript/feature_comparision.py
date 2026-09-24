"""Staged-gene recovery benchmark: how many of the staged genes does each
method find?

The metric is a plain count -- of the `n_staged` staged genes, how many appear
in each method's top-N panel. Chance is n_top * n_staged / ngenes.

SCENARIOS
  null      True null by LABEL PERMUTATION: the count matrix is untouched and
            the group labels are shuffled, so no gene is differential while
            every gene keeps its real library coupling, dispersion and
            gene-gene correlation. No staged genes are inserted; the expected
            recovery is zero for every method.

  trimodal  group_2 splits evenly at centre-sep, centre, centre+sep, so mean
            and median again match group_1 -- same spacing as bimodal, one more
            mode.

  bimodal   group_1 ~ Normal(centre, sd)                        one peak
            group_2 ~ 0.5*N(centre-sep, sd) + 0.5*N(centre+sep, sd)  two peaks
            The modes sit symmetrically about `centre`, so group_2's mean AND
            median equal group_1's on the count scale. Only the SHAPE differs.

  skew      group_1 right-tailed, group_2 left-tailed, mirror images with the
            same mean on the count scale. Again only the shape differs, this
            time in the direction of the tail rather than the number of peaks.

  coexpression
            Same marginal distribution in both groups, but the staged genes are
            positively co-dependent only within group_2. This is a dependence-
            only benchmark, not a simple gene-wise DE benchmark.

STAGING CONVENTION. Staged rates are multiplied by each cell's library-size
factor, exactly as scsim generates every background gene. This matters: staged
genes drawn at a fixed rate become the most library-anticorrelated genes in the
panel (measured r = -0.89 against +0.13 for background), and any method that
uses library size in its clustering recovers them for a reason unrelated to the
scenario. Under the overwrite convention recursieve recovered 8.2/10 staged
genes from a NULL dataset; under label permutation it sits at chance.

METHODS
  Wilcoxon, TTest   imported from features_comparison.py, byte-identical to it
  LogReg            scanpy logistic-regression ranking
  recursieve        recursive RF + GMM relabeling
  recursieve_unique recursieve's unique_gene_panel

Trial i uses sim_seed+i, so trials are independent datasets.

Outputs (to --out-dir, default cwd):
  feature_comparison_wide_counts.csv
  feature_comparison_wide_heatmap.{png,pdf}

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


# =========================================================================
# 1. Datasets
# =========================================================================

def _background(sim_seed, ncells, ngenes):
	"""scsim background with the parameters used by features_comparison.py."""
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


def _cell_scale(adata):
	"""Per-cell library-size multiplier, from the background totals."""
	lib = np.asarray(adata.X).sum(axis=1)
	return (lib / float(np.median(lib))).reshape(-1, 1)


def _to_counts(values):
	return np.clip(np.rint(values), 0, None).astype(np.int64)


def _match_block_totals(pattern, totals, rng=None):
	"""Rescale a staged pattern so each cell's staged-block total is unchanged.

	`totals[i]` is the sum scsim itself produced across the staged genes for
	cell i. The pattern carries the SHAPE we want; this rescales each cell's row
	so it sums to exactly that total, then rounds by largest remainder so the
	sum is preserved as integers.
	"""
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
	"""Per-cell sum across the staged genes, as scsim generated them."""
	return np.asarray(adata.X)[:, :n_staged].sum(axis=1).astype(np.int64)


def _stage_coexpressed(rng, n1, n2, n_staged):
	"""Co-expression staging — matched marginals, correlated only in group_2."""
	g1_block = rng.negative_binomial(5, 0.5, size=(n1, n_staged))
	g2_block = rng.negative_binomial(5, 0.5, size=(n2, n_staged))
	shared_order = np.argsort(rng.standard_normal(n2))
	for j in range(n_staged):
		g2_block[:, j] = np.sort(g2_block[:, j])[shared_order]
	return g1_block, g2_block


def build_null(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
				ngenes=2000, **_):
	"""True null: labels are shuffled, but no staged genes are inserted."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	adata.obs["group"] = pd.Categorical(
		rng.permutation(adata.obs["group"].values))
	return adata, []


def build_bimodal(sim_seed=32, stage_seed=0, n_staged=10, ncells=2000,
				  ngenes=2000, center=8.0, sd=1.5, sep=4.0, **_):
	"""group_1 normal about `center`; group_2 bimodal about the SAME center."""
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
	"""group_1 normal about `center`; group_2 TRIMODAL about the same centre."""
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
					   ngenes=2000, library_scaled=False, **_):
	"""Gene-gene dependence in group_2 with matched marginal distributions."""
	adata = _background(sim_seed, ncells, ngenes)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())

	if not library_scaled:
		g1, g2 = _stage_coexpressed(rng, n1, n2, n_staged)
	else:
		cs = _cell_scale(adata)
		cs1, cs2 = cs[m1], cs[~m1]
		lat1 = rng.negative_binomial(5, 0.5, size=(n1, n_staged)).astype(float)
		lat2 = rng.negative_binomial(5, 0.5, size=(n2, n_staged)).astype(float)
		order = np.argsort(rng.standard_normal(n2))
		for j_ in range(n_staged):
			lat2[:, j_] = np.sort(lat2[:, j_])[order]
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
	"""group_1 RIGHT-tailed, group_2 LEFT-tailed, matched mean."""
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
			"coexpression":
				lambda **kw: build_coexpression(library_scaled=True, **kw)}


# =========================================================================
# 2. Methods
# =========================================================================

def _normalized(adata):
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	return a


def _filter_ranked_genes(adata, method, n_top=50, pval_cutoff=0.05):
	"""Rank genes and keep only those meeting a p-value threshold when available.

	Scanpy exposes p-values for Wilcoxon and T-test, but not for logreg ranking.
	When no p-value array exists, we fall back to the ranked list without a
	statistical filter rather than crashing the benchmark.
	"""
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
	return list(model.genes)[:n_top], list(model.unique_gene_panel)[:n_top]


# =========================================================================
# 3. Metric + staging report
# =========================================================================

def count_recovered(panel, staged):
	"""The whole metric: how many staged genes are in this panel."""
	return int(sum(1 for g in panel if g in set(staged)))


def report_staging(adata, staged, scen, seed):
	"""Realised centres of the staged block, on counts and on log1p."""
	if not staged:
		print(f"  [{scen} s{seed}] true null: no staged genes inserted", flush=True)
		return
	raw = np.asarray(adata[:, staged].X)
	lg = np.asarray(_normalized(adata)[:, staged].X)
	g2 = (adata.obs["group"] == "group_2").values
	off = ~np.eye(len(staged), dtype=bool)
	r1 = np.corrcoef(lg[~g2].T)[off].mean()
	r2 = np.corrcoef(lg[g2].T)[off].mean()
	print(f"  [{scen} s{seed}] counts mean {raw[~g2].mean():6.2f}/{raw[g2].mean():6.2f}"
		  f"  median {np.median(raw[~g2]):5.1f}/{np.median(raw[g2]):5.1f}"
		  f"  | log1p mean {lg[~g2].mean():.3f}/{lg[g2].mean():.3f}"
		  f"  | pairwise r {r1:+.3f}/{r2:+.3f}", flush=True)


# =========================================================================
# 4. Figure
# =========================================================================

_RAMP = ["#f4f7fb", "#dce7f3", "#bcd2e8", "#94b6d8", "#6b97c4",
		 "#4a79ac", "#325d8e", "#1f4370", "#132c4d"]
_INK, _MUTED, _SURFACE = "#1a1d21", "#5c636b", "#ffffff"


TITLE = "DE tests ignore distributional shifts"

SCENARIO_DESC = {
	"null":    "labels shuffled; no real difference",
	"bimodal": "g1 one peak, g2 two peaks, same centre",
	"trimodal": "g1 one peak, g2 three peaks, same centre",
	"skew":    "g1 right-tailed, g2 left-tailed, same mean",
	"coexpression": "same marginals; genes co-vary only in g2",
}
METHOD_DESC = [
	"Wilcoxon: ranks (location)      TTest: means (location)      "
	"LogReg: multivariate coefficients",
	"recursieve: recursive random forest + GMM relabeling",
]


def _normalize_scenario_labels(df):
	"""Preserve 'null' as a real scenario label instead of pandas NA."""
	df = df.copy()
	df["scenario"] = df["scenario"].astype("string")
	df["scenario"] = df["scenario"].fillna("null")
	df["scenario"] = df["scenario"].replace({"nan": "null", "NaN": "null", "None": "null", "": "null"})
	df["scenario"] = df["scenario"].astype(str)
	return df


def plot_heatmap(df, out_dir, n_staged, chance, n_trials, scenarios):
	"""One row per scenario, one column per method, coloured by the MEAN count."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap

	df = _normalize_scenario_labels(df)
	cols = [m for m in METHOD_ORDER if m in set(df["method"])]
	ordered_scenarios = [s for s in SCENARIO_ORDER if s in scenarios]
	missing = [s for s in scenarios if s not in ordered_scenarios]
	ordered_scenarios.extend(missing)
	piv = (df.pivot_table(index="scenario", columns="method",
						  values="n_recovered")[cols].reindex(ordered_scenarios, fill_value=0.0))
	M = np.nan_to_num(piv.to_numpy(dtype=float), nan=0.0)

	cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
	fig, ax = plt.subplots(figsize=(max(9.5, 1.15 * M.shape[1] + 2.6),
									 0.95 * M.shape[0] + 3.4))
	fig.subplots_adjust(left=0.30, right=0.94, top=0.77, bottom=0.2)
	fig.patch.set_facecolor(_SURFACE); ax.set_facecolor(_SURFACE)
	mesh = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=n_staged,
						 edgecolors=_SURFACE, linewidth=2)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = M[i, j]
			if np.isfinite(v):
				label = "0.0" if abs(v) < 1e-12 else f"{v:.1f}"
				ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
						fontsize=11,
						color=_SURFACE if v > n_staged * 0.55 else _INK)

	ax.set_xticks(np.arange(M.shape[1]) + 0.5)
	ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9, color=_INK)
	ax.set_yticks(np.arange(M.shape[0]) + 0.5)
	ax.set_yticklabels(
		[f"{i}\n{SCENARIO_DESC.get(i, '')}" for i in piv.index],
		fontsize=9, color=_INK)
	ax.invert_yaxis()
	for sp in ax.spines.values():
		sp.set_visible(False)
	ax.tick_params(length=0)

	cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
	cbar.set_label(f"mean genes recovered (of {n_staged})", fontsize=9, color=_INK)
	cbar.outline.set_visible(False)
	cbar.ax.tick_params(length=0, labelsize=8, colors=_MUTED)

	short_title = (
		f"{TITLE}\n"
		f"recovered of {n_staged}; mean over {n_trials} trials; "
		f"expected random overlap = {chance:.3f}"
	)
	fig.suptitle(short_title, fontsize=12, color=_INK, y=0.98, x=0.52, ha="center")
	fig.text(0.012, 0.012, "\n".join(METHOD_DESC), fontsize=7.5,
			 color=_MUTED, va="bottom", ha="left", linespacing=1.6)
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"feature_comparison_wide_heatmap.{ext}",
					dpi=300, facecolor=_SURFACE)
	plt.close(fig)


# =========================================================================
# 5. Run
# =========================================================================

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
					help="bimodal: distance from centre to each mode")
	ap.add_argument("--alpha", type=float, default=6.0,
					help="skew: skew-normal shape; group_1 +alpha, group_2 -alpha")
	ap.add_argument("--title", default=TITLE)
	ap.add_argument("--out-dir", default=".")
	ap.add_argument("--plot-only", action="store_true",
					help="skip benchmark recomputation and plot the saved CSV only")
	args = ap.parse_args()

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	bad = [s for s in scenarios if s not in BUILDERS]
	if bad:
		raise SystemExit(f"unknown scenarios {bad}; choose from {sorted(BUILDERS)}")

	out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
	csv = out_dir / "feature_comparison_wide_counts.csv"
	chance = args.n_top * args.n_staged / args.ngenes

	existing = pd.DataFrame()
	if csv.exists():
		existing = pd.read_csv(csv, keep_default_na=False, na_values=[], dtype={"scenario": "string"})
		existing = _normalize_scenario_labels(existing)
		existing = existing[existing["scenario"].astype(str).ne("")]
		if set(existing.columns) != {"scenario", "seed", "method", "n_recovered", "panel_size"}:
			existing = pd.DataFrame(columns=["scenario", "seed", "method", "n_recovered", "panel_size"])

	if args.plot_only:
		df = existing.copy()
		df = df[df["scenario"].astype(str).ne("")]
		df = df.reset_index(drop=True)
		df = _normalize_scenario_labels(df)
		if df.empty:
			raise SystemExit(f"No benchmark data found in {csv}; run without --plot-only to generate it first.")
		piv = df.pivot_table(index="scenario", columns="method", values="n_recovered")
		all_scenarios = [s for s in SCENARIO_ORDER if s in BUILDERS and s in set(piv.index) | set(scenarios)]
		all_scenarios.extend([s for s in dict.fromkeys(list(piv.index) + scenarios) if s not in all_scenarios and s in BUILDERS])
		print("\n" + "=" * 78)
		print(f"PLOTTING STORED DATA FROM {csv}")
		print("=" * 78)
		print(piv[[m for m in METHOD_ORDER if m in piv.columns]]
			  .reindex(all_scenarios).round(2).to_string())
		globals()["TITLE"] = args.title
		plot_heatmap(df, out_dir, args.n_staged, chance, args.seeds, all_scenarios)
		return

	rows = []
	for scen in scenarios:
		for seed in range(args.seeds):
			t0 = time.time()
			adata, staged = BUILDERS[scen](
				sim_seed=args.sim_seed + seed, stage_seed=seed,
				n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes,
				center=args.center, sd=args.sd, sep=args.sep, alpha=args.alpha)
			report_staging(adata, staged, scen, seed)

			panels = {
				"Wilcoxon": run_wilcoxon(adata, n_top=args.n_top,
											 pval_cutoff=args.pval_cutoff),
				"TTest": run_ttest(adata, n_top=args.n_top,
											 pval_cutoff=args.pval_cutoff),
				"LogReg": run_logreg(adata, n_top=args.n_top,
											 pval_cutoff=args.pval_cutoff),
			}
			rec, uniq = run_recursieve(adata, n_top=args.n_top,
											 max_iterations=args.max_iterations,
											 pval_cutoff=args.pval_cutoff)
			panels["recursieve"], panels["recursieve_unique"] = rec, uniq

			line = []
			for m in METHOD_ORDER:
				n = count_recovered(panels[m], staged)
				rows.append(dict(scenario=scen, seed=seed, method=m,
							n_recovered=n, panel_size=len(panels[m])))
				line.append(f"{m}={n}")
			print("     " + "  ".join(line) + f"   [{time.time()-t0:.0f}s]", flush=True)

	df_new = pd.DataFrame(rows)
	if not df_new.empty:
		df = pd.concat([existing, df_new], ignore_index=True)
		df = df[df["scenario"].astype(str).ne("")]
		df = df.drop_duplicates(subset=["scenario", "seed", "method"], keep="last")
		df = df.reset_index(drop=True)
	else:
		df = existing.copy()
		df = df[df["scenario"].astype(str).ne("")]
		df = df.reset_index(drop=True)
		df = _normalize_scenario_labels(df)

	df.to_csv(csv, index=False)

	print("\n" + "=" * 78)
	print(f"STAGED GENES RECOVERED (of {args.n_staged}; expected random overlap = {chance:.3f}; "
		  f"mean of n={args.seeds})")
	print("=" * 78)
	piv = df.pivot_table(index="scenario", columns="method", values="n_recovered")
	all_scenarios = [s for s in SCENARIO_ORDER if s in BUILDERS and s in set(piv.index) | set(scenarios)]
	all_scenarios.extend([s for s in dict.fromkeys(list(piv.index) + scenarios) if s not in all_scenarios and s in BUILDERS])
	print(piv[[m for m in METHOD_ORDER if m in piv.columns]]
		  .reindex(all_scenarios).round(2).to_string())

	globals()["TITLE"] = args.title
	plot_heatmap(df, out_dir, args.n_staged, chance, args.seeds, all_scenarios)


if __name__ == "__main__":
	main()
