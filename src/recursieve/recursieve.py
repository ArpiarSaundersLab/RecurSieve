import gc
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from AnnSQL import AnnSQL
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list


class recursieve:
	"""
	Iterative gene selection using random forest and GMM clustering.

	Recursively selects discriminative genes between two groups by training
	random forests and collapsing selected genes via GMM. Stops when label
	oscillation is detected or max iterations reached.

	Parameters
	----------
	adata : anndata.AnnData
		Annotated data matrix with cells in obs, genes in var.
	annsq_db : anndata.AnnData, optional
		Precomputed AnnSql database
	group1 : str
		Label of first group in field_name column.
	group2 : str
		Label of second group in field_name column.
	field_name : str, optional
		Column in obs containing group labels. Default is "sample".
	plots : bool, optional
		Generate visualization plots during iterations. Default is False.
	print_to_console : bool, optional
		Print selected genes to stdout. Default is False.
	max_iterations : int, optional
		Maximum number of selection iterations. Default is 100.
	additive : bool, optional
		If True, add new genes to panel. If False, use only latest gene.
		Default is True.
	flip_rate_percentage : float, optional
		Threshold for label oscillation detection (0 to 1). Stop when label
		flip rate drops below this. Default is 0.01.
	patience : int or None, optional
		Stop when the flip rate has not reached a new minimum for this many
		consecutive iterations (it has plateaued above flip_rate_percentage).
		None disables the rule. Default is 10.
	pval_cutoff : float, optional
		P-value threshold for differential expression filtering.
		Default is 0.05.
	logfc_cutoff : float, optional
		Absolute log2 fold-change threshold for differential expression
		filtering. Genes with |logFC| below this value are excluded.
		Default is 0.1.
	seed : int, optional
		Random seed for reproducibility (used in RF, train/test split, GMM).
		Default is 42.
	summary_method : str, optional
		Method to collapse selected genes: "mean" averages expression,
		"pca" uses first principal component. Default is "mean".
	n_estimators : int, optional
		Number of trees in random forest. Default is 300.
	n_jobs : int, optional
		Number of parallel jobs for RF. -1 uses all cores. Default is -1.
	max_depth : int or None, optional
		Maximum depth of each RF tree. Limits memorization of noise.
		None grows trees until leaves are pure. Default is 10.
	preprocess : bool, optional
		Run QC, normalization, and HVG selection. Set False when adata is
		already preprocessed (used internally by seed_stability).
		Default is True.
	n_seeds : int, optional
		Number of seeds used by seed_stability. Default is 10.
	run : bool, optional
		Run the full pipeline (fit) on construction. Set False to configure
		the object and call fit() later. Default is True.

	Attributes
	----------
	genes : list
		Genes selected in order of selection.
	acuracy_scores : list
		Accuracy on test set for each iteration.
	auc_scores : list
		AUC on test set for each iteration.
	oob_scores : list
		Out-of-bag accuracy of the RF for each iteration.
	flip_rates : list
		Label flip rate vs the previous iteration, from iteration 2 on.
	stop_reason : str
		Why selection stopped: "threshold", "oscillation", "patience", or
		"max_iterations".
	unique_gene_panel : list
		Genes in self.genes not in DE results (higher priority genes).

	gene_expression_log : OrderedDict
		Expression arrays for each selected gene.

	Examples
	--------
	>>> import anndata as ad
	>>> from recursieve import recursieve
	>>> adata = ad.read_h5ad("data.h5ad")
	>>> model = recursieve(adata, group1="ctrl", group2="treat",
	...                     field_name="condition", max_iterations=50)
	>>> print(model.unique_gene_panel[:5])
	"""
	def __init__(self, adata, group1, group2, field_name="sample", plots=False,
				 print_to_console=False, max_iterations=100, additive=True,
				 flip_rate_percentage=0.01, pval_cutoff=0.05, logfc_cutoff=0.1,
				 seed=42, summary_method="mean", n_estimators=300, n_jobs=-1,
				 annsql_db=None, max_depth=10, preprocess=True,
				 n_seeds=10, run=True, patience=10):
		# constructor args, reused by seed_stability to rerun the pipeline
		self._params = dict(
			group1=group1, group2=group2, field_name=field_name, plots=plots,
			print_to_console=print_to_console, max_iterations=max_iterations,
			additive=additive, flip_rate_percentage=flip_rate_percentage,
			pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, seed=seed,
			summary_method=summary_method, n_estimators=n_estimators,
			n_jobs=n_jobs, max_depth=max_depth, patience=patience,
		)
		self.n_seeds = n_seeds
		self.adata = adata
		self.annsql_db = annsql_db
		self.group1 = group1
		self.group2 = group2
		self.field_name = field_name
		self.plots = plots
		self.print_to_console = print_to_console
		self.max_iterations = max_iterations
		self.additive = additive
		self.flip_rate_percentage = flip_rate_percentage
		self.patience = patience
		self.pval_cutoff = pval_cutoff
		self.logfc_cutoff = logfc_cutoff
		self.seed = seed
		self.summary_method = summary_method
		self.gene_expression_log = OrderedDict()  # preserves selection order
		self.genes = []
		self.accuracy_scores = []
		self.auc_scores = []
		self.oob_scores = []
		self.rows = []
		self.label_history = []  # for oscillation detection

		self.rf_params = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"max_features": "sqrt",
			"min_samples_split": 2,
			"min_samples_leaf": 1,
			"class_weight": "balanced_subsample",
			"oob_score": True,
			"n_jobs": n_jobs,
		}

		self.preprocess = preprocess
		if run:
			self.fit()

	def fit(self):
		"""
		Run the full pipeline: preprocessing, DE, gene selection.

		Notes
		-----
		Called automatically in __init__ unless run=False.

		Examples
		--------
		>>> model = recursieve(adata, "ctrl", "treat", run=False)
		>>> model.fit()
		"""
		if self.preprocess:
			self.preprocessing()
		self.filter_groups()
		self.scanpy_de_original_groups()
		self.ensemble_learner()
		self.ensemble_recursion()
		self.unique_gene_panel = self._unique_gene_panel()
		return self

	def replace_var_names(self):
		"""
		Replace gene names with feature names from var table.

		Swaps var_names with values from var["feature_name"] and stores
		original names in var["ensembl_id"].

		Raises
		------
		KeyError
			If "feature_name" column is not in adata.var.

		Examples
		--------
		>>> model.replace_var_names()
		>>> print(model.adata.var.columns)
		"""
		if "feature_name" in self.adata.var.columns:
			self.adata.var["ensembl_id"] = self.adata.var_names.astype(str)
			self.adata.var_names = self.adata.var["feature_name"].astype(str)
			self.adata.var_names_make_unique()
		else:
			raise KeyError("feature_name not found in adata.var.")

	def preprocessing(self):
		"""
		Apply quality control and normalization to expression data.

		Filters mitochondrial, ribosomal, and hemoglobin genes. Calculates
		QC metrics, filters cells and genes by count thresholds, normalizes,
		log-transforms, and selects top 2000 highly variable genes. Densifies
		matrix if memory permits.

		Notes
		-----
		Modifies self.adata in place. Stores original counts in layers.

		Examples
		--------
		>>> model = recursieve(...)
		>>> model.preprocessing()  # called automatically in __init__
		"""

		#if annsql exists, but no anndata then open the db and make a anndata object
		if self.annsql_db is not None and self.adata is None:
			
			#open the annsql db and write to a temp.h5ad file
			AnnSQL(self.annsql_db).write_adata(filename='temp.h5ad')
			
			#open the temp.h5ad file and assign to self.adata
			self.adata = ad.read_h5ad('temp.h5ad')
			
			#delete the temp.h5ad file
			os.remove('temp.h5ad')


		self.adata.layers["counts"] = self.adata.X.copy()
		self.adata.var["mt"] = self.adata.var_names.str.upper().str.startswith("MT-")
		self.adata.var["ribo"] = self.adata.var_names.str.upper().str.match(r"^RPS|^RPL")
		self.adata.var["hb"] = self.adata.var_names.str.upper().str.startswith("HB")
		self.adata = self.adata[:, ~(self.adata.var["mt"] |
									 self.adata.var["ribo"] |
									 self.adata.var["hb"])].copy()
		percent_top = tuple(
			p for p in (50, 100, 200, 300)
			if p < self.adata.n_vars
		)
		sc.pp.calculate_qc_metrics(
			self.adata,
			qc_vars=["mt", "ribo", "hb"],
			inplace=True,
			log1p=True,
			percent_top=percent_top,
		)
		sc.pp.filter_cells(self.adata, min_genes=100)
		sc.pp.filter_genes(self.adata, min_cells=10)
		sc.pp.normalize_total(self.adata)
		sc.pp.log1p(self.adata)
		sc.pp.highly_variable_genes(self.adata, n_top_genes=2000, subset=True)

		# SPEEDUP: densify once if it fits, RF doesn't benefit from sparse
		# and we re-slice X many times
		if sp.issparse(self.adata.X):
			# only densify if reasonable size (< ~4GB float32)
			nbytes_dense = self.adata.n_obs * self.adata.n_vars * 4
			if nbytes_dense < 4e9:
				self.adata.X = self.adata.X.toarray()

	def filter_groups(self):
		"""
		Subset data to cells from the two specified groups.

		Keeps only cells with labels matching group1 or group2 in field_name.

		Raises
		------
		ValueError
			If either group has zero cells.

		Examples
		--------
		>>> model.filter_groups()  # called automatically in __init__
		"""
		mask1 = self.adata.obs[self.field_name] == self.group1
		mask2 = self.adata.obs[self.field_name] == self.group2
		n1, n2 = int(mask1.sum()), int(mask2.sum())
		if n1 == 0 or n2 == 0:
			available = self.adata.obs[self.field_name].unique().tolist()
			raise ValueError(
				f"group1='{self.group1}' has {n1} cells, group2='{self.group2}' "
				f"has {n2} cells in field '{self.field_name}'. "
				f"Available: {available}"
			)
		self.adata = self.adata[mask1 | mask2].copy()
		gc.collect()

	def _get_X_dense(self, adata=None):
		"""
		Return dense expression matrix.

		If input is sparse, convert to dense array. Otherwise return as is.

		Parameters
		----------
		adata : anndata.AnnData, optional
			Use this object instead of self.adata. Default is None.

		Returns
		-------
		np.ndarray
			Dense expression matrix of shape (n_obs, n_vars).

		Examples
		--------
		>>> X = model._get_X_dense()
		>>> X.shape
		(5000, 2000)
		"""
		a = adata if adata is not None else self.adata
		X = a.X
		if sp.issparse(X):
			X = X.toarray()
		return X

	def ensemble_learner(self):
		"""
		Train random forest and select top gene by importance.

		Fits RF on all remaining genes, computes feature importances, and
		selects the highest-ranked gene. Calculates accuracy and AUC on test
		set. Stores gene expression and metrics.

		Notes
		-----
		Runs on first call to select initial gene. Called automatically in
		__init__ and ensemble_recursion().

		Examples
		--------
		>>> model.ensemble_learner()  # called automatically
		>>> print(model.genes[0])
		"""
		X = self._get_X_dense()
		y = self.adata.obs[self.field_name].to_numpy()

		# remove already-used genes (none on first call, kept for parity)
		if len(self.genes) > 0:
			genes_mask = np.asarray(~self.adata.var_names.isin(self.genes))
			X = X[:, genes_mask]
			self.adata = self.adata[:, genes_mask].copy()
			gc.collect()

		self.rf, self.df, acc, auc = self._fit_rf(
			X, y, self.adata.var_names, self.rf_params, self.seed, self.group2
		)
		top_gene = self.df["genes"].iloc[0]
		self.genes.append(top_gene)

		if self.plots:
			self.ensemble_learner_plots()
		if self.print_to_console:
			print(top_gene)

		self.accuracy_scores.append(acc)
		self.auc_scores.append(auc)
		self.oob_scores.append(float(getattr(self.rf, "oob_score_", float("nan"))))

		# store expression as 1D float32 array (memory + downstream simplicity)
		expr = self.adata[:, top_gene].X
		if sp.issparse(expr):
			expr = expr.toarray()
		self.gene_expression_log[top_gene] = np.asarray(expr).ravel().astype(np.float32)

	@staticmethod
	def _fit_rf(X, y, var_names, rf_params, seed, positive):
		"""
		Fit RF on a stratified 70/30 split and rank genes by importance.

		Parameters
		----------
		X : np.ndarray
			Expression matrix of shape (n_cells, n_genes).
		y : array-like
			Labels of length n_cells.
		var_names : array-like
			Gene names matching the columns of X.
		rf_params : dict
			Keyword arguments for RandomForestClassifier.
		seed : int
			Random seed for the split and RF.
		positive : str
			Label treated as the positive class for AUC.

		Returns
		-------
		tuple
			(rf, importance_df, accuracy, auc) where importance_df has
			columns "genes" and "importance" sorted descending. auc is NaN
			if y does not have exactly two classes.
		"""
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.3, random_state=seed, stratify=y
		)
		rf = RandomForestClassifier(**rf_params, random_state=seed)
		rf.fit(X_train, y_train)
		preds = rf.predict(X_test)

		df = pd.DataFrame({
			"genes": var_names,
			"importance": rf.feature_importances_
		}).sort_values("importance", ascending=False).reset_index(drop=True)

		auc = float("nan")
		if len(set(y)) == 2:
			y_test_bin = (y_test == positive).astype(int)
			proba = rf.predict_proba(X_test)[:, list(rf.classes_).index(positive)]
			auc = roc_auc_score(y_test_bin, proba)
		return rf, df, accuracy_score(y_test, preds), auc

	@staticmethod
	def _first_step_summary(X, y, var_names, rf_params, seed, positive):
		"""Fit the first-iteration RF; return (top_gene, auc, oob, margin).

		margin is the importance gap between the top two genes.
		"""
		rf, df, _, auc = recursieve._fit_rf(X, y, var_names, rf_params, seed, positive)
		imp = df["importance"].to_numpy()
		margin = float(imp[0] - imp[1]) if len(imp) > 1 else float("nan")
		return df["genes"].iloc[0], auc, float(rf.oob_score_), margin

	def _panel_from_first_gene(self, X, gene, **overrides):
		"""
		Run the recursion starting from a given first gene.

		Parameters
		----------
		X : np.ndarray
			Dense expression matrix matching self.adata.
		gene : str
			First gene of the panel.
		**overrides
			Constructor arguments to override (e.g. seed, max_iterations).

		Returns
		-------
		list
			Selected genes in order, starting with gene.
		"""
		params = dict(self._params, plots=False, print_to_console=False,
					  preprocess=False, run=False)
		params.update(overrides)
		return self._build_panel(X, self.adata.obs[["log1p_total_counts"]],
								 self.adata.var_names, params, gene)

	@staticmethod
	def _auto_parallel(cores, data_bytes):
		"""
		Default number of parallel workers, limited by cores and free memory.

		Uses about one worker per 4 cores, but no more than fit in 70% of
		currently available memory, assuming each worker needs about ten
		times the size of the data it copies. Results do not depend on the
		number of workers.

		Parameters
		----------
		cores : int
			Number of CPU cores.
		data_bytes : int
			Size of the arrays each worker copies.

		Returns
		-------
		int
			Number of workers (at least 1).
		"""
		n = max(1, cores // 4)
		try:
			with open("/proc/meminfo") as f:
				info = dict(line.split(":", 1) for line in f)
			available = int(info["MemAvailable"].split()[0]) * 1024
			n = min(n, int(0.7 * available // max(1, 10 * data_bytes)))
		except (OSError, KeyError, ValueError):
			pass  # no /proc/meminfo (non-Linux): keep the core-based default
		return max(1, n)

	@staticmethod
	def _build_panel(X, obs, var_names, params, gene):
		"""Run the recursion from a given first gene (picklable for workers)."""
		base = ad.AnnData(X=X, obs=obs.copy(), var=pd.DataFrame(index=var_names))
		model = recursieve(base, **params)
		model.genes = [gene]
		model.gene_expression_log[gene] = X[:, var_names.get_loc(gene)].astype(np.float32)
		model.ensemble_recursion()
		genes = list(model.genes)
		del model, base
		gc.collect()
		return genes

	def ensemble_learner_plots(self):
		"""
		Generate three plots of top gene expression.

		Creates scatterplot (UMI vs expression), violin plot (group vs expr),
		and histogram (expression density) for the highest-ranked gene.

		Notes
		-----
		Called only if plots=True during initialization.

		Examples
		--------
		>>> model = recursieve(..., plots=True)
		>>> model.ensemble_learner_plots()
		"""
		top = self.df["genes"].iloc[0]
		expr = self.adata[:, top].X
		if sp.issparse(expr):
			expr = expr.toarray()
		expr = np.asarray(expr).ravel()

		sns.scatterplot(x=self.adata.obs.log1p_total_counts.values.flatten(),
						y=expr, hue=self.adata.obs[self.field_name], s=4)
		plt.xlabel("log1p_total_counts"); plt.ylabel(top); plt.show(); plt.clf()

		sns.violinplot(x=self.adata.obs[self.field_name], y=expr)
		plt.xlabel(self.field_name); plt.ylabel(top); plt.show(); plt.clf()

		sns.histplot(x=expr, hue=self.adata.obs[self.field_name].values,
					 element="step", stat="density", common_norm=False)
		plt.xlabel(top); plt.ylabel("Density"); plt.show(); plt.clf()

	def _summarize_selected(self):
		"""
		Collapse selected gene expressions to single 1D axis.

		Uses PCA (first PC) or mean aggregation based on summary_method.
		PCA handles co-expressed genes better by reducing redundancy.

		Returns
		-------
		np.ndarray
			1D float32 array of length n_cells.

		Examples
		--------
		>>> summary = model._summarize_selected()
		>>> summary.shape
		(5000,)
		"""
		mat = np.column_stack([
			v for k, v in self.gene_expression_log.items() if k != "summed"
		])  # cells x n_selected

		if self.summary_method == "pca" and mat.shape[1] >= 2:
			# first PC handles redundancy: co-expressed genes contribute once
			pc = PCA(n_components=1, random_state=self.seed).fit_transform(
				StandardScaler().fit_transform(mat)
			).ravel()
			return pc.astype(np.float32)
		# default: mean (stable scale across iterations)
		return mat.mean(axis=1).astype(np.float32)

	def cluster_infectivity_gmm(self, gene=None):
		"""
		Cluster cells using Gaussian Mixture Model on gene expression.

		Fits 2-component GMM on log-transformed UMI and gene expression.
		Assigns higher expression cluster label "1". Stores labels in obs.

		Parameters
		----------
		gene : str, optional
			Unused (kept for API compatibility). Uses latest gene from self.genes.

		Returns
		-------
		tuple
			(gmm_model, labels) where labels is (n_cells,) int array.

		Notes
		-----
		If additive=True, uses summary of all selected genes. Otherwise
		uses expression of single latest gene.

		Examples
		--------
		>>> gmm, labels = model.cluster_infectivity_gmm()
		>>> print(labels.shape)
		(5000,)
		"""
		gene = self.genes[-1]

		if self.additive:
			expr = self._summarize_selected().reshape(-1, 1)
			self.gene_expression_log["summed"] = expr.ravel()
			self.adata.obs["rf_summed"] = expr.ravel()
		else:
			e = self.adata[:, gene].X
			if sp.issparse(e):
				e = e.toarray()
			expr = np.asarray(e).reshape(-1, 1)

		counts = self.adata.obs["log1p_total_counts"].to_numpy().reshape(-1, 1)
		X = StandardScaler().fit_transform(np.hstack([counts, expr]))

		gmm = GaussianMixture(n_components=2, random_state=self.seed)
		labels = gmm.fit_predict(X)

		# label cluster with higher expr as "1"
		means = [expr[labels == k].mean() for k in [0, 1]]
		hi = int(np.argmax(means))
		labels = (labels == hi).astype(int)

		self.adata.obs[f"gmm_{gene}"] = labels.astype(str)
		self.gmm = gmm

		if self.plots:
			sns.scatterplot(x=counts.ravel(), y=expr.ravel(),
							hue=self.adata.obs[f"gmm_{gene}"], s=3)
			plt.xlabel("log1p_total_counts"); plt.ylabel(gene)
			plt.title(f"GMM Clusters for {gene}"); plt.show(); plt.clf()

		if self.print_to_console:
			print(gene)
		return gmm, labels

	@staticmethod
	def _flip_rate(a, b):
		"""
		Compute minimum label disagreement rate between two binary arrays.

		Returns the minimum of disagreement rate and disagreement rate
		with inverted labels (to account for label swap). Used to detect
		when clustering is stable.

		Parameters
		----------
		a : array-like
			First binary array.
		b : array-like
			Second binary array.

		Returns
		-------
		float
			Minimum disagreement rate between 0 and 1.

		Examples
		--------
		>>> a = np.array([0, 1, 1, 0])
		>>> b = np.array([0, 1, 1, 0])
		>>> recursieve._flip_rate(a, b)
		0.0
		"""
		a = np.asarray(a).astype(int).ravel()
		b = np.asarray(b).astype(int).ravel()
		return float(min(np.mean(a != b), np.mean((1 - a) != b)))

	def _check_oscillation(self, labels, window=3):
		"""
		Detect 2-cycle oscillation in cluster labels.

		Compares current labels to labels from 2 iterations ago. If the
		flip rate to t-2 is low but flip rate to t-1 is high, the algorithm
		is oscillating between two states. Maintains a sliding window of
		label history.

		Parameters
		----------
		labels : array-like
			Current cluster labels.
		window : int, optional
			Number of iterations to track. Default is 3.

		Returns
		-------
		bool
			True if 2-cycle oscillation detected, False otherwise.

		Examples
		--------
		>>> labels = np.array([0, 1, 1, 0])
		>>> is_osc = model._check_oscillation(labels)
		"""
		self.label_history.append(labels.copy())
		if len(self.label_history) < window:
			return False
		fr_prev = self._flip_rate(labels, self.label_history[-2])
		fr_prev2 = self._flip_rate(labels, self.label_history[-3])
		# trim history
		if len(self.label_history) > window:
			self.label_history.pop(0)
		return (fr_prev2 < self.flip_rate_percentage) and (fr_prev > self.flip_rate_percentage)

	def ensemble_recursion(self, iteration=1):
		"""
		Iteratively select genes using RF and GMM clustering.

		Runs main loop that selects genes one at a time. Each iteration:
		1. Cluster cells with GMM on selected genes or single latest gene
		2. Train RF on remaining genes using GMM labels
		3. Select top gene by importance
		Stops when the label flip rate drops below flip_rate_percentage, a
		2-cycle oscillation is detected, or the flip rate plateaus for
		patience iterations.

		Parameters
		----------
		iteration : int, optional
			Starting iteration number. Default is 1.

		Notes
		-----
		Called automatically in __init__. Updates self.genes, self.auc_scores,
		self.accuracy_scores, and self.gene_expression_log.

		Examples
		--------
		>>> model.ensemble_recursion()  # called automatically
		>>> len(model.genes)
		25
		"""
		self.prev_labels = None
		self.flip_rates = []
		self.stop_reason = "max_iterations"
		best_fr, stall = float("inf"), 0
		while iteration <= self.max_iterations:
			print(f"Iteration {iteration}")
			gene = self.genes[-1]
			_, labels = self.cluster_infectivity_gmm()
			y = self.adata.obs[f"gmm_{gene}"].to_numpy()

			if self.prev_labels is not None:
				fr = self._flip_rate(labels, self.prev_labels)
				print(f"flip_rate={fr:.4f}")
				self.flip_rates.append(fr)
				if fr < self.flip_rate_percentage:
					print("stopping: <threshold label change vs previous iteration")
					self.stop_reason = "threshold"
					break
				if self._check_oscillation(labels):
					print("stopping: 2-cycle oscillation detected")
					self.stop_reason = "oscillation"
					break
				if fr < best_fr:
					best_fr, stall = fr, 0
				else:
					stall += 1
				if self.patience is not None and stall >= self.patience:
					print(f"stopping: no new flip-rate minimum in {self.patience} iterations")
					self.stop_reason = "patience"
					break

			self.prev_labels = labels.copy()

			# RF on remaining genes
			genes_mask = ~self.adata.var_names.isin(self.genes)
			genes_mask = np.asarray(genes_mask)
			X = self._get_X_dense()[:, genes_mask]
			remaining_names = self.adata.var_names[genes_mask]

			rf, self.df, acc, auc = self._fit_rf(
				X, y, remaining_names, self.rf_params, self.seed, "1"
			)
			self.auc_scores.append(auc)
			self.accuracy_scores.append(acc)
			self.oob_scores.append(float(getattr(rf, "oob_score_", float("nan"))))

			top_gene = self.df["genes"].iloc[0]
			self.genes.append(top_gene)

			expr = self.adata[:, top_gene].X
			if sp.issparse(expr):
				expr = expr.toarray()
			self.gene_expression_log[top_gene] = np.asarray(expr).ravel().astype(np.float32)

			if self.print_to_console:
				print(top_gene)
			iteration += 1

	def _filter_de_df(self, df):
		"""Filter DE results by p-value and absolute log2 fold-change thresholds."""
		col_logfc = next((c for c in ["logfoldchanges", "logfc"] if c in df.columns), None)
		col_pval = next((c for c in ["pvals", "pval"] if c in df.columns), None)
		if not all([col_logfc, col_pval]):
			raise KeyError(f"DE columns missing. Got: {list(df.columns)}")

		filtered = df[(df[col_pval] < self.pval_cutoff) &
				 (df[col_logfc].abs() >= self.logfc_cutoff)].copy()
		return filtered.sort_values(col_pval, kind="mergesort")

	def scanpy_de_original_groups(self, n_top=None, method="wilcoxon"):
		"""
		Compute differential expression reference for gene filtering.

		Ranks genes by differential expression between group1 and group2
		using scanpy. Filters by p-value and absolute log fold-change cutoffs
		and stores results in self.de_dict. Both upregulated and downregulated
		genes are included.

		Parameters
		----------
		n_top : int, optional
			Cap number of DE genes (after p-value and logFC filtering).
			Default is None (no cap).
		method : str, optional
			Differential expression test ("wilcoxon", "t-test", etc).
			Default is "wilcoxon".

		Notes
		-----
		Called automatically in __init__. Stores DE info in self.de_dict
		as OrderedDict with gene names as keys.

		Examples
		--------
		>>> model.scanpy_de_original_groups(n_top=500, method="wilcoxon")
		>>> print(len(model.de_dict))
		500
		"""
		sc.tl.rank_genes_groups(
			self.adata, groupby=self.field_name,
			groups=[self.group2], reference=self.group1,
			method=method, n_genes=self.adata.n_vars, use_raw=False,
		)
		df = sc.get.rank_genes_groups_df(self.adata, group=self.group2)

		col_logfc = next((c for c in ["logfoldchanges", "logfc"] if c in df.columns), None)
		col_pval = next((c for c in ["pvals", "pval"] if c in df.columns), None)
		col_padj = next((c for c in ["pvals_adj", "pval_adj"] if c in df.columns), None)
		col_score = next((c for c in ["scores", "score"] if c in df.columns), None)
		if not all([col_logfc, col_pval, col_padj]):
			raise KeyError(f"DE columns missing. Got: {list(df.columns)}")

		df = self._filter_de_df(df)
		if n_top is not None:
			df = df.head(int(n_top))
		self.de_dict = OrderedDict()
		for _, row in df.iterrows():
			self.de_dict[row["names"]] = {
				"logfc": float(row[col_logfc]),
				"pval": float(row[col_pval]),
				"pval_adj": float(row[col_padj]),
				"score": float(row[col_score]) if col_score else None,
			}

	def intersect_genes_and_de(self, top_n_genes=None, keep_order="genes", return_df=True):
		"""
		Find genes in both selected list and DE results.

		Returns intersection of self.genes and self.de_dict keys, optionally
		merged with DE statistics. Can order by RF rank or DE rank.

		Parameters
		----------
		top_n_genes : int, optional
			Use only first N genes from self.genes. Default is None (all).
		keep_order : str, optional
			"genes" orders by RF rank, "de" orders by DE rank.
			Default is "genes".
		return_df : bool, optional
			If True, return (list, DataFrame). If False, return list only.
			Default is True.

		Returns
		-------
		tuple or list
			If return_df=True: (gene_list, dataframe with DE stats).
			If return_df=False: gene_list only.

		Examples
		--------
		>>> hits, df = model.intersect_genes_and_de(top_n_genes=50)
		>>> print(df.columns)
		['gene', 'rf_rank', 'logfc', 'pval', 'pval_adj', 'score']
		"""
		genes_list = list(self.genes)
		if top_n_genes is not None:
			genes_list = genes_list[:int(top_n_genes)]
		de_genes = list(self.de_dict.keys()) if hasattr(self, "de_dict") else []
		de_set = set(de_genes)

		if keep_order == "de":
			intersection = [g for g in de_genes if g in set(genes_list)]
		else:
			intersection = [g for g in genes_list if g in de_set]

		if not return_df:
			return intersection

		rf_rank = {g: i for i, g in enumerate(genes_list)}
		self.rows = []
		for g in intersection:
			s = self.de_dict.get(g, {})
			self.rows.append({
				"gene": g, "rf_rank": rf_rank.get(g, np.nan),
				"logfc": s.get("logfc", np.nan), "pval": s.get("pval", np.nan),
				"pval_adj": s.get("pval_adj", np.nan), "score": s.get("score", np.nan),
			})
		# Columns are declared so an empty intersection still yields a frame with
		# the right schema — otherwise pd.DataFrame([]) has no columns and the
		# sort raises KeyError. Empty is a legitimate result: it means no selected
		# gene was DE-significant, which is exactly what unique_gene_panel is for.
		cols = ["gene", "rf_rank", "logfc", "pval", "pval_adj", "score"]
		merged_df = (pd.DataFrame(self.rows, columns=cols)
					 .sort_values("rf_rank").reset_index(drop=True))
		return intersection, merged_df

	def _unique_gene_panel(self):
		"""
		Extract high-priority genes not in differential expression results.

		Returns genes selected by RF that were NOT found to be significantly
		differentially expressed. These are novel candidates. Preserves
		RF selection order.

		Returns
		-------
		list
			Genes in self.genes but not in self.de_dict, in RF order.

		Examples
		--------
		>>> unique = model._unique_gene_panel()
		>>> print(unique[:5])
		['GENE1', 'GENE3', 'GENE5', ...]
		"""
		hits, _ = self.intersect_genes_and_de()
		hits_set = set(hits)
		# preserve order of self.genes (RF rank) instead of returning a set
		return [g for g in self.genes if g not in hits_set]

	def seed_stability(self, n_seeds=10, max_iterations=None, n_jobs=None,
					   n_parallel=None):
		"""
		Measure how stable the selected panel is across random seeds.

		Reruns gene selection on the true labels with seeds self.seed,
		self.seed + 1, ..., which changes the train/test split, RF
		randomness, and GMM initialization. Reports first-gene consistency,
		per-gene selection frequency, and panel overlap.

		Parameters
		----------
		n_seeds : int, optional
			Number of seeds, starting at self.seed. Default is None (uses
			self.n_seeds).
		max_iterations : int, optional
			Override max_iterations for the reruns. Default is None (same
			as the observed run).
		n_jobs : int, optional
			Number of first-iteration fits run in parallel (each RF uses one
			core). Default is None (uses the n_jobs given at construction).
		n_parallel : int, optional
			Number of seed panels built at once; the CPU cores are split
			between them. Results do not depend on it. Default is None
			(about one panel per 4 cores, fewer if memory is short).

		Returns
		-------
		dict
			"runs" : pandas.DataFrame with seed, first_gene, auc, oob,
			margin (importance gap between the top two genes), n_genes, and
			jaccard (overlap with the observed panel) per seed.
			"gene_frequency" : pandas.DataFrame with the fraction of seeds
			whose panel contains each gene, and whether the gene is in the
			observed panel, sorted by frequency.
			"mean_pairwise_jaccard" : float, mean panel overlap across all
			pairs of seeds.
			"panels" : pandas.DataFrame (seed, rank, gene) listing every
			seed's panel, one row per gene.

		Notes
		-----
		The observed run is reused for self.seed. Each other seed costs one
		full recursion (preprocessing and DE are skipped). Results are
		stored in self.seed_stability_results.

		Examples
		--------
		>>> res = model.seed_stability(n_seeds=20)
		>>> res["gene_frequency"].head()
		"""
		if n_seeds is None:
			n_seeds = self.n_seeds
		if n_jobs is None:
			n_jobs = self._params["n_jobs"]
		if max_iterations is None:
			max_iterations = self.max_iterations
		seeds = [self.seed + i for i in range(n_seeds)]

		X = self._get_X_dense()
		y = self.adata.obs[self.field_name].to_numpy()
		rf_params = dict(self.rf_params, n_jobs=1)
		first = Parallel(n_jobs=n_jobs)(
			delayed(recursieve._first_step_summary)(
				X, y, self.adata.var_names, rf_params, s, self.group2)
			for s in seeds
		)

		# panels for each seed, built in parallel; the observed run is reused
		reuse = max_iterations == self.max_iterations
		todo = [(s, g) for s, (g, _, _, _) in zip(seeds, first)
				if not (reuse and s == self.seed)]
		cores = os.cpu_count() or 1
		if n_parallel is None:
			n_parallel = self._auto_parallel(cores, X.nbytes)
		n_parallel = max(1, min(n_parallel, len(todo) or 1))
		base_params = dict(self._params, plots=False, print_to_console=False,
						   preprocess=False, run=False, max_iterations=max_iterations,
						   n_jobs=max(1, cores // n_parallel))
		obs = self.adata.obs[["log1p_total_counts"]]
		for s, g in todo:
			print(f"Seed {s}: panel from first gene {g}")
		built = Parallel(n_jobs=n_parallel)(
			delayed(recursieve._build_panel)(
				X, obs, self.adata.var_names, dict(base_params, seed=s), g)
			for s, g in todo
		)
		built = dict(zip([s for s, _ in todo], built))
		panels = [list(self.genes) if s not in built else built[s] for s in seeds]

		def jaccard(a, b):
			a, b = set(a), set(b)
			return len(a & b) / len(a | b)

		runs = pd.DataFrame(first, columns=["first_gene", "auc", "oob", "margin"])
		runs.insert(0, "seed", seeds)
		runs["n_genes"] = [len(p) for p in panels]
		runs["jaccard"] = [jaccard(p, self.genes) for p in panels]

		all_genes = list(dict.fromkeys(g for p in panels for g in p))
		observed = set(self.genes)
		freq_df = pd.DataFrame({
			"gene": all_genes,
			"frequency": [sum(g in p for p in panels) / n_seeds for g in all_genes],
			"in_observed": [g in observed for g in all_genes],
		}).sort_values("frequency", ascending=False, kind="mergesort").reset_index(drop=True)

		pairs = [jaccard(panels[i], panels[j])
				 for i in range(n_seeds) for j in range(i + 1, n_seeds)]
		panels_df = pd.DataFrame(
			[(s, i + 1, g) for s, p in zip(seeds, panels) for i, g in enumerate(p)],
			columns=["seed", "rank", "gene"])
		self.seed_stability_results = {
			"runs": runs, "gene_frequency": freq_df,
			"mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else float("nan"),
			"panels": panels_df,
		}
		return self.seed_stability_results

	def coexpression_null(self, n_shuffles=25, max_iterations=None, n_parallel=None, verbose=0):
		"""
		Rerun selection on data with gene-gene co-expression destroyed.

		Every raw count is drawn from the pooled counts of all genes: the raw
		count matrix (layers["counts"]) is permuted across cells and genes
		together. Every gene then has the same (population) distribution, so
		no gene is favored by its own mean or variance, and no gene is
		related to any other gene. Each shuffled cell is then processed like
		a real one: library size (log1p_total_counts, used by the GMM) is
		recomputed from its shuffled counts, and counts are normalized to the
		median total and log1p-transformed. Group labels are permuted across
		cells, unrelated to everything else. The full algorithm, including
		the first iteration and GMM reclustering, is then run on each
		shuffled dataset. A real co-expression module should converge quickly
		and stably; shuffled data should not. Because shuffled genes are
		exchangeable, each gene is expected in a shuffled panel with
		probability (panel size / number of genes), and the recomputed
		library size favors no gene.

		Each shuffled run selects exactly as many genes as the observed panel
		(stopping rules off), so per-gene frequencies compare panels of equal
		length. Without this, shuffled runs stop after a few genes: with no
		co-expression the GMM labels follow the unchanging library-size
		column, the flip rate falls below the threshold, and short shuffled
		panels would make every observed gene look rarely selected.

		Parameters
		----------
		n_shuffles : int, optional
			Number of independently shuffled matrices. Default is 3.
		max_iterations : int, optional
			Override the number of iterations of the shuffled runs. Default
			is None (len(self.genes) - 1, so shuffled panels match the
			observed panel length).
		n_parallel : int, optional
			Number of shuffles run at once; the CPU cores are split between
			them. Default is None (about one shuffle per 4 cores, fewer if
			memory is short).
		verbose : int, optional
			joblib progress level; 10 or more reports every finished
			shuffle. Default is 0 (silent).

		Returns
		-------
		dict
			"runs" : pandas.DataFrame with one row for the observed run and
			one per shuffle: first_gene, first_auc, mean_later_auc (mean
			test AUC of iterations after the first), n_genes, stop_reason,
			min_flip_rate, final_flip_rate, and jaccard with the observed
			panel.
			"flip_rates" : pandas.DataFrame (run, iteration, flip_rate) for
			plotting convergence.
			"panels" : pandas.DataFrame (run, rank, gene) listing every
			panel, observed included.
			"pvalues" : pandas.DataFrame of module-level empirical p-values
			(see _null_pvalues).
			"gene_frequency" : pandas.DataFrame of per-gene shuffled
			frequency and p-value (see _null_gene_frequency).

		Notes
		-----
		Requires a fitted model with raw counts in layers["counts"] (stored
		by preprocessing). Shuffled library sizes sum the selected genes
		only, so their scale differs from the real ones; the GMM
		standardizes its inputs. Shuffles are independent and run in
		parallel, each with its own seed (seed, shuffle index), so results do
		not depend on n_parallel. OOB scoring is off in shuffled runs since
		no null statistic uses it. Differential expression is skipped for
		shuffled runs. Results are stored in self.coexpression_null_results.

		Examples
		--------
		>>> res = model.coexpression_null(n_shuffles=3)
		>>> res["runs"]
		"""
		if "counts" not in self.adata.layers:
			raise ValueError('coexpression_null needs raw counts in adata.layers["counts"]')
		X = self._get_X_dense(ad.AnnData(X=self.adata.layers["counts"]))
		obs = self.adata.obs[[self.field_name]].copy()
		# shuffled panels match the observed length: fixed iteration count,
		# threshold, oscillation, and patience stopping all switched off
		params = dict(self._params, plots=False, print_to_console=False,
					  preprocess=False, run=False, flip_rate_percentage=-1.0,
					  patience=None,
					  max_iterations=(max_iterations if max_iterations is not None
									  else len(self.genes) - 1))

		cores = os.cpu_count() or 1
		if n_parallel is None:
			n_parallel = self._auto_parallel(cores, X.nbytes)
		n_parallel = max(1, min(n_parallel, n_shuffles))
		params["n_jobs"] = max(1, cores // n_parallel)

		results = Parallel(n_jobs=n_parallel, verbose=verbose)(
			delayed(recursieve._run_null_shuffle)(
				X, obs, self.adata.var_names, params, (self.seed, k), self.genes)
			for k in range(1, n_shuffles + 1)
		)
		rows = [self._null_summary(self, "observed", self.genes)]
		curves = [("observed", i + 2, f) for i, f in enumerate(self.flip_rates)]
		panels = [("observed", i + 1, g) for i, g in enumerate(self.genes)]
		for k, (summary, flips, genes) in enumerate(results, start=1):
			name = f"shuffle_{k}"
			rows.append({"run": name, **summary})
			curves += [(name, i + 2, f) for i, f in enumerate(flips)]
			panels += [(name, i + 1, g) for i, g in enumerate(genes)]

		runs = pd.DataFrame(rows)
		panels = pd.DataFrame(panels, columns=["run", "rank", "gene"])
		self.coexpression_null_results = {
			"runs": runs,
			"flip_rates": pd.DataFrame(curves, columns=["run", "iteration", "flip_rate"]),
			"panels": panels,
			"pvalues": self._null_pvalues(runs),
			"gene_frequency": self._null_gene_frequency(panels),
		}
		return self.coexpression_null_results

	@staticmethod
	def _null_summary(model, name, observed_genes):
		"""Summary statistics of one run for the co-expression null table."""
		later = np.asarray(model.auc_scores[1:], dtype=float)
		fr = model.flip_rates
		panel, observed = set(model.genes), set(observed_genes)
		return {
			"run": name, "first_gene": model.genes[0],
			"first_auc": model.auc_scores[0],
			"mean_later_auc": float(np.nanmean(later)) if later.size else float("nan"),
			"n_genes": len(model.genes), "stop_reason": model.stop_reason,
			"min_flip_rate": min(fr) if fr else float("nan"),
			"final_flip_rate": fr[-1] if fr else float("nan"),
			"jaccard": len(panel & observed) / len(panel | observed),
		}

	@staticmethod
	def _pooled_null_data(counts, obs, rng):
		"""
		Draw every gene from the pooled counts, then reprocess each cell.

		Parameters
		----------
		counts : np.ndarray
			Raw count matrix (cells x genes).
		obs : pandas.DataFrame
			Cell-level label column(s), each permuted across cells.
		rng : numpy.random.Generator
			Random generator.

		Returns
		-------
		tuple
			(X, obs). X is the shuffled counts, normalized to the median
			total and log1p-transformed; obs holds the permuted labels and
			log1p_total_counts recomputed from the shuffled counts.
		"""
		pooled = rng.permuted(counts, axis=None).astype(np.float32, copy=False)
		totals = pooled.sum(axis=1)
		target = np.median(totals[totals > 0])  # as sc.pp.normalize_total
		scale = np.divide(target, totals, out=np.zeros_like(totals), where=totals > 0)
		X = np.log1p(pooled * scale[:, None])
		obs_s = obs.copy()
		for col in obs_s.columns:
			obs_s[col] = obs_s[col].to_numpy()[rng.permutation(len(obs_s))]
		obs_s["log1p_total_counts"] = np.log1p(totals)
		return X, obs_s

	@staticmethod
	def _run_null_shuffle(X, obs, var_names, params, seed, observed_genes):
		"""
		Build pooled null data (see _pooled_null_data) and run the algorithm once.

		Parameters
		----------
		X : np.ndarray
			Dense raw counts (cells x genes).
		obs : pandas.DataFrame
			obs with the group label column.
		var_names : pandas.Index
			Gene names matching the columns of X.
		params : dict
			Constructor arguments for the shuffled model (run=False).
		seed : int or tuple
			Seed for this shuffle's permutations.
		observed_genes : list of str
			Observed panel, for the overlap statistic.

		Returns
		-------
		tuple
			(summary dict without "run", flip rates, selected genes).
		"""
		Xs, obs_s = recursieve._pooled_null_data(X, obs, np.random.default_rng(seed))
		base = ad.AnnData(X=Xs, obs=obs_s, var=pd.DataFrame(index=var_names))
		model = recursieve(base, **params)
		model.rf_params["oob_score"] = False  # unused by null statistics
		model.ensemble_learner()
		model.ensemble_recursion()
		summary = recursieve._null_summary(model, None, observed_genes)
		summary.pop("run")
		return summary, list(model.flip_rates), list(model.genes)

	@staticmethod
	def _null_pvalues(runs):
		"""
		Empirical p-values of the observed run against shuffled runs.

		p = (1 + #shuffles at least as extreme as observed) / (1 + n_shuffles).
		Higher is more extreme for both AUCs. Flip rate is not tested: shuffled
		runs have no stopping rule, and their GMM labels follow library size.

		Parameters
		----------
		runs : pandas.DataFrame
			The "runs" table from coexpression_null.

		Returns
		-------
		pandas.DataFrame
			One row per statistic: observed, null_mean, null_min, null_max,
			n_shuffles, p_value.
		"""
		obs = runs[runs["run"] == "observed"].iloc[0]
		null = runs[runs["run"] != "observed"]
		n = len(null)
		rows = []
		for stat, greater in (("mean_later_auc", True), ("first_auc", True)):
			v = null[stat].to_numpy(dtype=float)
			extreme = (v >= obs[stat]) if greater else (v <= obs[stat])
			rows.append({
				"statistic": stat, "observed": float(obs[stat]),
				"null_mean": float(np.mean(v)), "null_min": float(np.min(v)),
				"null_max": float(np.max(v)), "n_shuffles": n,
				"p_value": (1 + int(extreme.sum())) / (1 + n),
			})
		return pd.DataFrame(rows)

	@staticmethod
	def _null_gene_frequency(panels):
		"""
		Per-gene frequency and p-value of observed panel genes in shuffled panels.

		A gene's p-value is (1 + #shuffled panels containing it) /
		(1 + n_shuffles); small values mean the gene is not selected without
		real co-expression.

		Parameters
		----------
		panels : pandas.DataFrame
			The "panels" table from coexpression_null.

		Returns
		-------
		pandas.DataFrame
			gene, rank, n_shuffled_panels, shuffled_frequency, p_value, in
			observed panel order.
		"""
		observed = panels[panels["run"] == "observed"].sort_values("rank")
		shuffled = panels[panels["run"] != "observed"]
		n = shuffled["run"].nunique()
		in_runs = shuffled.groupby("gene")["run"].nunique()
		freq = observed[["gene", "rank"]].reset_index(drop=True)
		freq["n_shuffled_panels"] = freq["gene"].map(in_runs).fillna(0).astype(int)
		freq["shuffled_frequency"] = freq["n_shuffled_panels"] / n if n else float("nan")
		freq["p_value"] = (1 + freq["n_shuffled_panels"]) / (1 + n)
		return freq

	@staticmethod
	def plot_common_expression(adata, genes_list):
		"""
		Visualizes Spearman correlation matrix of gene expressions.

		Filters gene list to those in adata, computes log-transformed
		expression correlations, hierarchically clusters, and plots as
		heatmap with Spearman correlation coefficient.

		Parameters
		----------
		adata : anndata.AnnData
			Annotated data with gene expressions.
		genes_list : list
			Gene names to correlate.

		Returns
		-------
		tuple
			(genes_reordered, correlation_matrix) where genes_reordered is
			clustered gene list and correlation_matrix is (n_genes, n_genes).
			Returns (genes, None) if fewer than 2 valid genes.

		Examples
		--------
		>>> genes_clustered, corr = model.plot_common_expression(
		...     model.adata, model.genes[:10])
		>>> print(corr.shape)
		(10, 10)
		"""
		genes = [g for g in genes_list if g in adata.var_names]
		if len(genes) < 2:
			print("Need at least 2 genes to plot correlation.")
			return genes, None
		X = adata[:, genes].X
		if sp.issparse(X):
			X = X.toarray()
		X = np.log1p(X)
		expr = pd.DataFrame(X, index=adata.obs_names, columns=genes)
		corr = expr.corr(method="spearman")
		Z = linkage(corr.values, method="average")
		order = leaves_list(Z)
		corr2 = corr.iloc[order, order]
		genes2 = corr2.columns.tolist()
		plt.figure(figsize=(0.35 * len(genes2) + 4, 0.35 * len(genes2) + 4))
		plt.imshow(corr2.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
		plt.xticks(range(len(genes2)), genes2, rotation=90)
		plt.yticks(range(len(genes2)), genes2)
		plt.colorbar(label="spearman r")
		plt.tight_layout(); plt.show()
		return genes2, corr2