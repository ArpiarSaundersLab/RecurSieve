"""Multi-dataset staged gene recovery benchmark (standalone, for HPC).

Simulates N datasets per scenario, runs Wilcoxon, t-test and logistic
regression (scanpy) once per dataset and RecurSieve with S seeds per dataset,
and counts how many of the 10 staged genes each method's top 50 panel recovers.
DE panels keep genes with Benjamini-Hochberg adjusted p < 0.05.

Dataset d uses sim_seed + d and stage_seed + d; RecurSieve seed i uses
rs_seed + i.

Scenarios:
  null          labels shuffled, no staged genes
  bimodal       group_2 has two modes around the group_1 mean
  trimodal      group_2 has three modes around the group_1 mean
  skew          group_1 right-tailed, group_2 left-tailed, same mean
  coexpression  same marginals, staged genes correlated only in group_2

The CSV is saved after every dataset and scenario, and a rerun skips the
blocks that are already complete, so an interrupted job can be resubmitted.

Chunk mode (to use many nodes): --make-todo splits the unfinished work into
chunks of --seed-chunk seeds (default 5) and writes them to todo_blocks.csv.
Array task i of N (--block-index i --n-tasks N) runs chunks i, i+N, ... and
writes each to parts/block_d<dataset>_<scenario>_s<first>-<last>.csv
(run_feature_blocks.sh). --plot-only merges these with everything else.

Every simulated dataset is checked before use: if the staged genes' per-cell
totals differ between groups (Welch t-test p < 1e-4), the run stops.

Outputs (to --out-dir):
  feature_comparison_wide_multi_counts.csv
  feature_comparison_wide_multi_heatmap_adj.png / .pdf
    (mean of the dataset means; SEM across dataset means)

Requires: recursieve, scanpy, anndata, numpy, pandas, scipy, matplotlib.

Usage:
  python feature_comparison_hpc.py --datasets 25 --seeds 25 --out-dir results
  python feature_comparison_hpc.py --out-dir results --plot-only
  python feature_comparison_hpc.py --out-dir results --make-todo
  python feature_comparison_hpc.py --out-dir results --n-tasks 100 --block-index 3
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import skewnorm, ttest_ind

import recursieve


# ---------------------------------------------------------------------------
# scsim single-cell count simulator (Kotliar et al., github.com/dylkot/scsim),
# copied unchanged from manuscript/scsim.py so this script is self-contained.
# ---------------------------------------------------------------------------
class scsim:
    def __init__(self, ngenes=10000, ncells=100, seed=757578,
                 mean_rate=.3, mean_shape=.6, libloc=11, libscale=0.2,
                expoutprob=.05, expoutloc=4, expoutscale=0.5, ngroups=1,
                diffexpprob=.1, diffexpdownprob=.5,
                diffexploc=.1, diffexpscale=.4, bcv_dispersion=.1,
                bcv_dof=60, ndoublets=0, groupprob=None,
                nproggenes=None, progdownprob=None, progdeloc=None,
                progdescale=None, proggoups=None, progcellfrac=None,
                minprogusage=.2, maxprogusage=.8):

        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.ngroups = ngroups
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof
        self.ndoublets = ndoublets
        self.init_ncells = ncells+ndoublets
        self.nproggenes=nproggenes
        self.progdownprob=progdownprob
        self.progdeloc=progdeloc
        self.progdescale=progdescale
        self.proggoups=proggoups
        self.progcellfrac = progcellfrac
        self.minprogusage = minprogusage
        self.maxprogusage = maxprogusage

        if groupprob is None:
            self.groupprob = [1/float(self.ngroups)]*self.ngroups
        elif (len(groupprob) == self.ngroups) & (np.abs(np.sum(groupprob) - 1) < (10**-6)):
            self.groupprob = groupprob
        else:
            sys.exit('Invalid groupprob input')


    def simulate(self):
        np.random.seed(self.seed)
        print('Simulating cells')
        self.cellparams = self.get_cell_params()
        print('Simulating gene params')
        self.geneparams = self.get_gene_params()

        if (self.nproggenes is not None) and (self.nproggenes > 0):
            print('Simulating program')
            self.simulate_program()

        print('Simulating DE')
        self.sim_group_DE()

        print('Simulating cell-gene means')
        self.cellgenemean = self.get_cell_gene_means()
        if self.ndoublets > 0:
            print('Simulating doublets')
            self.simulate_doublets()

        print('Adjusting means')
        self.adjust_means_bcv()
        print('Simulating counts')
        self.simulate_counts()

    def simulate_counts(self):
        '''Sample read counts for each gene x cell from Poisson distribution
        using the variance-trend adjusted updatedmean value'''
        self.counts = pd.DataFrame(np.random.poisson(lam=self.updatedmean),
                                   index=self.cellnames, columns=self.genenames)

    def adjust_means_bcv(self):
        '''Adjust cellgenemean to follow a mean-variance trend relationship'''
        self.bcv = self.bcv_dispersion + (1 / np.sqrt(self.cellgenemean))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv*np.sqrt(self.bcv_dof / chisamp)
        self.updatedmean = np.random.gamma(shape=1/(self.bcv**2),
                                           scale=self.cellgenemean*(self.bcv**2))
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updatedmean = pd.DataFrame(self.updatedmean, index=self.cellnames,
                                        columns=self.genenames)


    def simulate_doublets(self):
        ## Select doublet cells and determine the second cell to merge with
        d_ind = sorted(np.random.choice(self.ncells, self.ndoublets,
                                        replace=False))
        d_ind = ['Cell%d' % (x+1) for x in d_ind]
        self.cellparams['is_doublet'] = False
        self.cellparams.loc[d_ind, 'is_doublet'] = True
        extraind = self.cellparams.index[-self.ndoublets:]
        group2 = self.cellparams.ix[extraind, 'group'].values
        self.cellparams['group2'] = -1
        self.cellparams.loc[d_ind, 'group2'] = group2

        ## update the cell-gene means for the doublets while preserving the
        ## same library size
        dmean = self.cellgenemean.loc[d_ind,:].values
        dmultiplier = .5 / dmean.sum(axis=1)
        dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])
        omean = self.cellgenemean.loc[extraind,:].values
        omultiplier = .5 / omean.sum(axis=1)
        omean = np.multiply(omean, omultiplier[:,np.newaxis])
        newmean = dmean + omean
        libsize = self.cellparams.loc[d_ind, 'libsize'].values
        newmean = np.multiply(newmean, libsize[:,np.newaxis])
        self.cellgenemean.loc[d_ind,:] = newmean
        ## remove extra doublet cells from the data structures
        self.cellgenemean.drop(extraind, axis=0, inplace=True)
        self.cellparams.drop(extraind, axis=0, inplace=True)
        self.cellnames = self.cellnames[0:self.ncells]


    def get_cell_gene_means(self):
        '''Calculate each gene's mean expression for each cell while adjusting
        for the library size'''


        group_genemean = self.geneparams.loc[:,[x for x in self.geneparams.columns if ('_genemean' in x) and ('group' in x)]].T.astype(float)
        group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)
        ind = self.cellparams['group'].apply(lambda x: 'group%d_genemean' % x)

        if self.nproggenes == 0:
            cellgenemean = group_genemean.loc[ind,:].astype(float)
            cellgenemean.index = self.cellparams.index
        else:
            noprogcells = self.cellparams['has_program']==False
            hasprogcells = self.cellparams['has_program']==True

            print('   - Getting mean for activity program carrying cells')
            progcellmean = group_genemean.loc[ind[hasprogcells], :]
            progcellmean.index = ind.index[hasprogcells]
            progcellmean = progcellmean.multiply(1-self.cellparams.loc[hasprogcells, 'program_usage'], axis=0)

            progmean = self.geneparams.loc[:,['prog_genemean']]
            progmean = progmean.div(progmean.sum(axis=0), axis=1)
            progusage = self.cellparams.loc[progcellmean.index, ['program_usage']]
            progusage.columns = ['prog_genemean']
            progcellmean += progusage.dot(progmean.T)
            progcellmean = progcellmean.astype(float)

            print('   - Getting mean for non activity program carrying cells')
            noprogcellmean = group_genemean.loc[ind[noprogcells],:]
            noprogcellmean.index = ind.index[noprogcells]

            cellgenemean = pd.concat([noprogcellmean, progcellmean], axis=0)

            del(progcellmean, noprogcellmean)

            cellgenemean = cellgenemean.reindex(index=self.cellparams.index)

        print('   - Normalizing by cell libsize')
        normfac = (self.cellparams['libsize'] / cellgenemean.sum(axis=1)).values
        for col in cellgenemean.columns:
            cellgenemean[col] = cellgenemean[col].values*normfac
        #cellgenemean = cellgenemean.multiply(normfac, axis=0).astype(float)
        return(cellgenemean)


    def get_gene_params(self):
        '''Sample each genes mean expression from a gamma distribution as
        well as identifying outlier genes with expression drawn from a
        log-normal distribution'''
        basegenemean = np.random.gamma(shape=self.mean_shape,
                                       scale=1./self.mean_rate,
                                       size=self.ngenes)

        is_outlier = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.expoutprob,1-self.expoutprob])
        outlier_ratio = np.ones(shape=self.ngenes)
        outliers = np.random.lognormal(mean=self.expoutloc,
                                       sigma=self.expoutscale,
                                       size=is_outlier.sum())
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers*median
        self.genenames = ['Gene%d' % i for i in range(1, self.ngenes+1)]
        geneparams = pd.DataFrame([basegenemean, is_outlier, outlier_ratio, gene_mean],
                                  index=['BaseGeneMean', 'is_outlier', 'outlier_ratio', 'gene_mean'],
                                 columns=self.genenames).T
        return(geneparams)


    def get_cell_params(self):
        '''Sample cell group identities and library sizes'''
        groupid = self.simulate_groups()
        libsize = np.random.lognormal(mean=self.libloc, sigma=self.libscale,
                                      size=self.init_ncells)
        self.cellnames = ['Cell%d' % i for i in range(1, self.init_ncells+1)]
        cellparams = pd.DataFrame([groupid, libsize],
                                  index=['group', 'libsize'],
                                  columns=self.cellnames).T
        cellparams['group'] = cellparams['group'].astype(int)
        return(cellparams)


    def simulate_program(self):
        ## Simulate the program gene expression
        self.geneparams['prog_gene'] = False
        proggenes = self.geneparams.index[-self.nproggenes:]
        self.geneparams.loc[proggenes, 'prog_gene'] = True
        DEratio = np.random.lognormal(mean=self.progdeloc,
                                          sigma=self.progdescale,
                                          size=self.nproggenes)
        DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
        is_downregulated = np.random.choice([True, False],
                                            size=len(DEratio),
                                            p=[self.progdownprob,
                                            1-self.progdownprob])
        DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
        all_DE_ratio = np.ones(self.ngenes)
        all_DE_ratio[-self.nproggenes:] = DEratio
        prog_mean = self.geneparams['gene_mean']*all_DE_ratio
        self.geneparams['prog_genemean'] = prog_mean

        ## Assign the program to cells
        self.cellparams['has_program'] = False
        if self.proggoups is None:
            ## The program is active in all cell types
            self.proggoups = np.arange(1, self.ngroups+1)

        self.cellparams.loc[:, 'program_usage'] = 0
        for g in self.proggoups:
            groupcells = self.cellparams.index[self.cellparams['group']==g]
            hasprog = np.random.choice([True, False], size=len(groupcells),
                                      p=[self.progcellfrac,
                                      1-self.progcellfrac])
            self.cellparams.loc[groupcells[hasprog], 'has_program'] = True
            usages = np.random.uniform(low=self.minprogusage,
                                       high=self.maxprogusage,
                                       size=len(groupcells[hasprog]))
            self.cellparams.loc[groupcells[hasprog], 'program_usage'] = usages




    def simulate_groups(self):
        '''Sample cell group identities from a categorical distriubtion'''
        groupid = np.random.choice(np.arange(1, self.ngroups+1),
                                       size=self.init_ncells, p=self.groupprob)
        self.groups = np.unique(groupid)
        return(groupid)


    def sim_group_DE(self):
        '''Sample differentially expressed genes and the DE factor for each
        cell-type group'''
        groups = self.cellparams['group'].unique()
        if self.nproggenes>0:
            proggene = self.geneparams['prog_gene'].values
        else:
            proggene = np.array([False]*self.geneparams.shape[0])

        for group in self.groups:
            isDE = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.diffexpprob,1-self.diffexpprob])
            isDE[proggene] = False # Program genes shouldn't be differentially expressed between groups
            DEratio = np.random.lognormal(mean=self.diffexploc,
                                          sigma=self.diffexpscale,
                                          size=isDE.sum())
            DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
            is_downregulated = np.random.choice([True, False],
                                            size=len(DEratio),
                                            p=[self.diffexpdownprob,1-self.diffexpdownprob])
            DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            group_mean = self.geneparams['gene_mean']*all_DE_ratio

            deratiocol = 'group%d_DEratio' % group
            groupmeancol = 'group%d_genemean' % group
            self.geneparams[deratiocol] = all_DE_ratio
            self.geneparams[groupmeancol] = group_mean


# ---------------------------------------------------------------------------
# benchmark (same code as manuscript/feature_comparision.py)
# ---------------------------------------------------------------------------
METHOD_ORDER = ["Wilcoxon", "TTest", "LogReg", "recursieve"]
SCENARIO_ORDER = ["null", "bimodal", "trimodal", "skew", "coexpression"]


def _background(sim_seed, ncells, ngenes):
	"""Simulated background counts."""
	sim = scsim(
		ngenes=ngenes, ncells=ncells, ngroups=2,
		libloc=7.64, libscale=0.78,
		mean_rate=7.68, mean_shape=0.34,
		expoutprob=0.00286, expoutloc=6.15, expoutscale=0.49,
		# almost no background group DE: staged genes keep the totals of the genes they
		# replace, so a DE gene there would leak a mean difference into the block
		diffexpprob=0.001, diffexpdownprob=0., diffexploc=1.0, diffexpscale=1.0,
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


def check_staging(adata, staged, p_min=1e-4):
	"""Stop if the staged block's per-cell totals differ between the groups.

	Staged genes keep each cell's total of the genes they replace, so a group
	difference in those totals would give every staged gene a real mean
	difference. Welch t-test on log1p totals; clean backgrounds give p > 0.009.
	"""
	if not staged:
		return
	totals = np.asarray(adata.X)[:, :len(staged)].sum(axis=1)
	g2 = (adata.obs["group"] == "group_2").to_numpy()
	p = ttest_ind(np.log1p(totals[g2]), np.log1p(totals[~g2]), equal_var=False).pvalue
	if p < p_min:
		ratio = totals[g2].mean() / totals[~g2].mean()
		raise SystemExit(f"staged block differs between groups (group_2/group_1 total "
						 f"{ratio:.2f}, p = {p:.2g}); staged genes would carry a mean difference")


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
	"""Top ranked genes with p < pval_cutoff, as (raw p panel, adjusted p panel).

	logreg has no p-values, so both panels are its top n_top genes.
	"""
	a = _normalized(adata)
	sc.tl.rank_genes_groups(a, groupby="group", groups=["group_2"],
							reference="group_1", method=method, use_raw=False)
	rankings = a.uns["rank_genes_groups"]
	names = np.asarray(rankings["names"]["group_2"])
	panels = []
	for key in ("pvals", "pvals_adj"):
		if key not in rankings:
			panels.append(names[:n_top].tolist())
			continue
		pvals = np.asarray(rankings[key]["group_2"], dtype=float)
		mask = np.isfinite(pvals) & (pvals < pval_cutoff)
		panels.append(names[mask].tolist()[:n_top])
	return tuple(panels)


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
SUBTITLE_ADJ = "(Simulated, adjusted p)"
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


def correction_note(pval_cutoff):
	"""Figure note on how the DE panels were filtered."""
	return (f"Wilcoxon and t-test panels: genes with Benjamini\u2013Hochberg adjusted "
			f"p < {pval_cutoff:g} (scanpy rank_genes_groups). LogReg has no p-values "
			"(top genes by score). RecurSieve does not filter on p-values.")


def plot_heatmap(df, out_dir, n_staged, scenarios, value="n_recovered",
				 subtitle=SUBTITLE, suffix="", tag="", note=None):
	"""Mean genes recovered per scenario and method.

	SEM is across seeds for one dataset, or across per-dataset means for several.
	"""
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
	n_ds = df["dataset"].nunique()
	n_seeds = int(df.groupby(["scenario", "method", "dataset"]).size().max())
	if n_ds > 1:
		sem_note = (f"Values are mean \u00b1 SEM across {n_ds} dataset means "
					f"({n_seeds} RecurSieve seeds each).")
		df = df.groupby(["scenario", "method", "dataset"], as_index=False)[value].mean()
	else:
		sem_note = (f"Values are mean \u00b1 SEM across {n_seeds} RecurSieve seeds on one "
					"dataset; DE methods are deterministic, so their SEM is 0.")
	note = f"{sem_note} {note}" if note else sem_note
	piv = (df.pivot_table(index="scenario", columns="method",
						  values=value)[cols].reindex(ordered_scenarios, fill_value=0.0))
	M = np.nan_to_num(piv.to_numpy(dtype=float), nan=0.0)
	sem = (df.pivot_table(index="scenario", columns="method", values=value,
						  aggfunc="sem")[cols].reindex(ordered_scenarios))
	S = np.nan_to_num(sem.to_numpy(dtype=float), nan=0.0)

	cmap = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
	fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * M.shape[1] + 2.4),
									 0.85 * M.shape[0] + 1.9))
	fig.subplots_adjust(left=0.22, right=0.86, top=0.88, bottom=0.29)
	fig.patch.set_facecolor(_SURFACE); ax.set_facecolor(_SURFACE)
	mesh = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=n_staged,
						 edgecolors=_SURFACE, linewidth=2)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			v = M[i, j]
			if np.isfinite(v):
				label = "0.0" if abs(v) < 1e-12 else f"{v:.1f}"
				color = _SURFACE if v > n_staged * 0.55 else _INK
				ax.text(j + 0.5, i + 0.42, label, ha="center", va="center",
						fontsize=8, color=color)
				ax.text(j + 0.5, i + 0.68, f"\u00b1{S[i, j]:.2f}", ha="center",
						va="center", fontsize=6, color=color)

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

	_mixed_text(fig, [(TITLE + " ", {"fontweight": "bold"}), (subtitle, {"fontstyle": "italic"})],
				0.5, 1.05, ax.transAxes, fontsize=14, color=_INK)
	fig.text(0.5, 0.02, textwrap.fill(note, 125), ha="center", va="bottom", fontsize=8,
			 color=_MUTED)
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"feature_comparison_wide{tag}_heatmap{suffix}.{ext}",
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
	ap.add_argument("--datasets", type=int, default=25, help="simulated datasets per scenario")
	ap.add_argument("--seeds", type=int, default=25, help="RecurSieve seeds per dataset")
	ap.add_argument("--first-dataset", type=int, default=0)
	ap.add_argument("--scenarios", default="null,bimodal,trimodal,skew,coexpression")
	ap.add_argument("--n-staged", type=int, default=10)
	ap.add_argument("--n-top", type=int, default=50, help="panel size per method")
	ap.add_argument("--pval-cutoff", type=float, default=0.05,
					help="BH adjusted p cutoff for the DE methods")
	ap.add_argument("--ncells", type=int, default=2000)
	ap.add_argument("--ngenes", type=int, default=2000)
	ap.add_argument("--sim-seed", type=int, default=32, help="dataset d uses sim_seed+d")
	ap.add_argument("--stage-seed", type=int, default=0, help="dataset d uses stage_seed+d")
	ap.add_argument("--rs-seed", type=int, default=42, help="seed i uses rs_seed+i")
	ap.add_argument("--max-iterations", type=int, default=50)
	ap.add_argument("--center", type=float, default=8.0)
	ap.add_argument("--sd", type=float, default=1.5)
	ap.add_argument("--sep", type=float, default=4.0,
					help="bimodal/trimodal: distance from center to each mode")
	ap.add_argument("--alpha", type=float, default=6.0,
					help="skew: skew-normal shape; group_1 +alpha, group_2 -alpha")
	ap.add_argument("--out-dir", default="feature_comparison_results")
	ap.add_argument("--plot-only", action="store_true",
					help="merge any array part files into the CSV and plot; run nothing")
	ap.add_argument("--n-tasks", type=int, default=1,
					help="number of array tasks sharing the chunks (with --block-index)")
	ap.add_argument("--make-todo", action="store_true",
					help="list the unfinished chunks of seeds in todo_blocks.csv")
	ap.add_argument("--seed-chunk", type=int, default=5,
					help="seeds per chunk for --make-todo")
	ap.add_argument("--block-index", type=int, default=None,
					help="with --n-tasks N, run chunks i, i+N, ... of todo_blocks.csv "
						 "(e.g. $SLURM_ARRAY_TASK_ID)")
	args = ap.parse_args()

	scenarios = [x.strip() for x in args.scenarios.split(",")]
	bad = [x for x in scenarios if x not in BUILDERS]
	if bad:
		raise SystemExit(f"unknown scenarios {bad}; choose from {sorted(BUILDERS)}")

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	csv = out_dir / "feature_comparison_wide_multi_counts.csv"
	parts_dir = out_dir / "parts"
	keys = ["dataset", "scenario", "seed", "method"]

	def read_counts(path):
		if not path.exists():
			return pd.DataFrame(columns=keys + ["n_recovered", "panel_size",
												"n_recovered_adj", "panel_size_adj"])
		return _normalize_scenario_labels(pd.read_csv(path, keep_default_na=False, na_values=[]))

	def tidy(df):
		df = df.copy()
		df["dataset"] = df["dataset"].astype(int)
		df["seed"] = df["seed"].astype(int)
		return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

	def complete(df, ds, scen):
		rs = df[(df["dataset"].astype(int) == ds) & (df["scenario"] == scen)
				& (df["method"] == "recursieve")]
		return rs["seed"].astype(int).isin(range(args.seeds)).sum() == args.seeds

	def run_block(ds, scen, seeds=None):
		"""Run one dataset x scenario block, or some of its seeds: DE methods once,
		RecurSieve per seed."""
		seeds = range(args.seeds) if seeds is None else seeds
		t_block = time.time()
		adata, staged = BUILDERS[scen](
			sim_seed=args.sim_seed + ds, stage_seed=args.stage_seed + ds,
			n_staged=args.n_staged, ncells=args.ncells, ngenes=args.ngenes,
			center=args.center, sd=args.sd, sep=args.sep, alpha=args.alpha)
		check_staging(adata, staged)
		# DE methods are deterministic: one run per dataset, reused for every seed
		# each entry is (raw p panel, adjusted p panel)
		de_panels = {
			"Wilcoxon": run_wilcoxon(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
			"TTest": run_ttest(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
			"LogReg": run_logreg(adata, n_top=args.n_top, pval_cutoff=args.pval_cutoff),
		}
		rows = []
		for seed in seeds:
			rs = run_recursieve(adata, n_top=args.n_top, seed=args.rs_seed + seed,
								max_iterations=args.max_iterations,
								pval_cutoff=args.pval_cutoff)
			for m in METHOD_ORDER:
				if m == "recursieve":
					# recursieve does not filter on p, so raw and adjusted match
					n = count_recovered(rs, staged)
					n, size, n_adj, size_adj = n, len(rs), n, len(rs)
				else:
					raw, adj = de_panels[m]
					n, size = count_recovered(raw, staged), len(raw)
					n_adj, size_adj = count_recovered(adj, staged), len(adj)
				rows.append(dict(dataset=ds, scenario=scen, seed=seed, method=m,
								 n_recovered=n, panel_size=size,
								 n_recovered_adj=n_adj, panel_size_adj=size_adj))
		rs_mean = np.mean([r["n_recovered_adj"] for r in rows if r["method"] == "recursieve"])
		print(f"[d{ds} {scen} seeds {seeds[0]}-{seeds[-1]}] RecurSieve mean {rs_mean:.2f}  "
			  f"[{time.time() - t_block:.0f}s]", flush=True)
		return rows

	todo_csv = out_dir / "todo_blocks.csv"
	if args.make_todo:
		# unfinished chunks of seeds, from the main CSV and every part file
		parts = sorted(parts_dir.glob("*.csv"))
		saved = pd.concat([read_counts(csv)] + [read_counts(f) for f in parts], ignore_index=True)
		rs = saved[saved["method"] == "recursieve"]
		have = set(zip(rs["dataset"].astype(int), rs["scenario"], rs["seed"].astype(int)))
		datasets = range(args.first_dataset, args.first_dataset + args.datasets)
		todo = []
		for d in datasets:
			for sc_ in scenarios:
				for start in range(0, args.seeds, args.seed_chunk):
					end = min(start + args.seed_chunk, args.seeds)
					if any((d, sc_, s_) not in have for s_ in range(start, end)):
						todo.append((d, sc_, start, end))
		pd.DataFrame(todo, columns=["dataset", "scenario", "seed_start", "seed_end"]).to_csv(
			todo_csv, index=False)
		print(f"{len(todo)} chunks of up to {args.seed_chunk} seeds still to run, listed in {todo_csv}")
		return

	if args.block_index is not None:
		# array task i runs chunks i, i + n_tasks, i + 2 * n_tasks, ... of the list
		# written by --make-todo, each into its own file
		todo = pd.read_csv(todo_csv, keep_default_na=False, na_values=[])
		mine = todo.iloc[args.block_index::args.n_tasks]
		parts_dir.mkdir(exist_ok=True)
		print(f"Task {args.block_index + 1}/{args.n_tasks}: {len(mine)} chunks", flush=True)
		for r in mine.itertuples(index=False):
			ds, scen = int(r.dataset), str(r.scenario)
			seeds = range(int(r.seed_start), int(r.seed_end))
			out_csv = parts_dir / f"block_d{ds:02d}_{scen}_s{seeds[0]:02d}-{seeds[-1]:02d}.csv"
			if out_csv.exists():
				print(f"[d{ds} {scen} seeds {seeds[0]}-{seeds[-1]}] already done, skipping")
				continue
			print(f"Block: dataset {ds}, {scen}, seeds {seeds[0]}-{seeds[-1]} -> {out_csv}",
				  flush=True)
			tidy(pd.DataFrame(run_block(ds, scen, seeds))).to_csv(out_csv, index=False)
		print("Task finished; run with --plot-only to merge the parts and plot.")
		return

	if args.plot_only:
		# merge the array tasks' part files into the main CSV, then plot
		parts = sorted(parts_dir.glob("*.csv"))
		df = pd.concat([read_counts(csv)] + [read_counts(f) for f in parts], ignore_index=True)
		if df.empty:
			raise SystemExit(f"No benchmark data in {csv} or {parts_dir}")
		df = tidy(df)
		df.to_csv(csv, index=False)
		print(f"Merged {len(parts)} part files into {csv}")
	else:
		# one job runs every dataset in turn, skipping blocks already in the CSV
		df = read_counts(csv)
		datasets = range(args.first_dataset, args.first_dataset + args.datasets)
		for ds in datasets:
			for scen in scenarios:
				if complete(df, ds, scen):
					print(f"[d{ds} {scen}] already done, skipping", flush=True)
					continue
				rows = run_block(ds, scen)
				# save after each dataset and scenario so a crash loses little
				df = tidy(pd.concat([df, pd.DataFrame(rows)], ignore_index=True))
				df.to_csv(csv, index=False)

	if df.empty:
		raise SystemExit(f"No benchmark data in {csv}")
	order = ordered_scenarios(df, scenarios)
	n_ds = df["dataset"].nunique()
	value = "n_recovered_adj"
	subtitle = SUBTITLE_ADJ.replace(")", f", {n_ds} datasets)" if n_ds > 1 else ")")
	piv = df.pivot_table(index="scenario", columns="method", values=value)
	print(piv[[m for m in METHOD_ORDER if m in piv.columns]].reindex(order).round(2).to_string())
	plot_heatmap(df, out_dir, args.n_staged, order, value=value, subtitle=subtitle,
				 suffix="_adj", tag="_multi", note=correction_note(args.pval_cutoff))
	print(f"Wrote {csv} and {out_dir / 'feature_comparison_wide_multi_heatmap_adj.png'}")

if __name__ == "__main__":
	main()
