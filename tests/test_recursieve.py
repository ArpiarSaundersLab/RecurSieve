import unittest

import numpy as np
import pandas as pd
import anndata as ad

import recursieve


class TestRecursieve(unittest.TestCase):
    def _make_small_dataset(self):
        rng = np.random.default_rng(0)
        n_group_1 = 80
        n_group_2 = 80
        n_genes = 400

        X = rng.poisson(lam=2.0, size=(n_group_1 + n_group_2, n_genes))
        X[:n_group_1, 0] += 10
        X[:n_group_1, 1] += 8
        X[n_group_1:, 0] += 1
        X[n_group_1:, 1] += 1
        X[n_group_1:, 2] += 12
        X[:n_group_1, 2] += 1

        obs = pd.DataFrame({
            "group": ["group_1"] * n_group_1 + ["group_2"] * n_group_2,
            "sample": ["s1"] * (n_group_1 + n_group_2),
        })
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_package_import_and_class_exists(self):
        self.assertTrue(hasattr(recursieve, "recursieve"))
        self.assertTrue(callable(recursieve.recursieve))

    def test_model_runs_on_small_dataset(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=3,
            plots=False,
            print_to_console=False,
            seed=42,
            n_estimators=50,
            n_jobs=1,
        )

        self.assertIsNotNone(model)
        self.assertGreater(len(model.genes), 0)
        self.assertGreaterEqual(len(model.accuracy_scores), 1)
        self.assertGreaterEqual(len(model.auc_scores), 1)
        self.assertIsInstance(model.unique_gene_panel, list)
        self.assertTrue(all(g in adata.var_names for g in model.genes))

    def test_missing_group_raises_value_error(self):
        adata = self._make_small_dataset().copy()
        adata = adata[adata.obs["group"] == "group_1"].copy()

        with self.assertRaises(ValueError):
            recursieve.recursieve(
                adata=adata,
                group1="group_1",
                group2="group_2",
                field_name="group",
                max_iterations=2,
                plots=False,
                print_to_console=False,
                seed=0,
                n_estimators=20,
                n_jobs=1,
            )

    def test_intersect_genes_and_de_returns_expected_shapes(self):
        adata = self._make_small_dataset()
        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=2,
            plots=False,
            print_to_console=False,
            seed=7,
            n_estimators=25,
            n_jobs=1,
        )

        hits, df = model.intersect_genes_and_de(top_n_genes=min(5, len(model.genes)))
        self.assertIsInstance(hits, list)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(set(df.columns).issuperset({"gene", "rf_rank", "logfc"}))

    def test_logfc_cutoff_parameter_filters_de_results(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=1,
            plots=False,
            print_to_console=False,
            seed=42,
            n_estimators=20,
            n_jobs=1,
            logfc_cutoff=0.1,
        )

        self.assertTrue(hasattr(model, "logfc_cutoff"))
        self.assertEqual(model.logfc_cutoff, 0.1)

        df = pd.DataFrame({
            "names": ["Gene0", "Gene1", "Gene2", "Gene3"],
            "logfoldchanges": [0.05, 0.11, -0.2, 0.0],
            "pvals": [0.01, 0.01, 0.01, 0.01],
            "pvals_adj": [0.02, 0.02, 0.02, 0.02],
            "scores": [1.0, 1.0, 1.0, 1.0],
        })

        filtered = model._filter_de_df(df)
        self.assertEqual(filtered["names"].tolist(), ["Gene1", "Gene2"])


    def test_oob_scores_recorded(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=2,
            seed=42,
            n_estimators=30,
            n_jobs=1,
            max_depth=5,
        )

        self.assertEqual(model.rf_params["max_depth"], 5)
        self.assertEqual(len(model.oob_scores), len(model.auc_scores))
        self.assertTrue(all(0 <= o <= 1 for o in model.oob_scores))

    def test_seed_stability(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=2,
            seed=42,
            n_estimators=30,
            n_jobs=1,
            max_depth=5,
        )

        res = model.seed_stability(n_seeds=3, n_parallel=1)
        runs = res["runs"]
        self.assertEqual(runs["seed"].tolist(), [42, 43, 44])
        # observed seed reproduces the observed panel
        self.assertEqual(runs["first_gene"].iloc[0], model.genes[0])
        self.assertEqual(runs["jaccard"].iloc[0], 1.0)
        freq = res["gene_frequency"]
        self.assertTrue(((freq["frequency"] > 0) & (freq["frequency"] <= 1)).all())
        self.assertTrue(0 <= res["mean_pairwise_jaccard"] <= 1)
        observed_panel = res["panels"].query("seed == 42").sort_values("rank")["gene"]
        self.assertEqual(observed_panel.tolist(), model.genes)
        # building panels in parallel must not change results
        par = model.seed_stability(n_seeds=3, n_parallel=2)
        for key in ("runs", "gene_frequency", "panels"):
            self.assertTrue(res[key].equals(par[key]), key)


    def test_patience_stops_no_later_than_without(self):
        adata = self._make_small_dataset()
        kw = dict(group1="group_1", group2="group_2", field_name="group",
                  max_iterations=8, seed=42, n_estimators=30, n_jobs=1,
                  max_depth=5, flip_rate_percentage=0.0)

        free = recursieve.recursieve(adata=adata.copy(), patience=None, **kw)
        patient = recursieve.recursieve(adata=adata.copy(), patience=1, **kw)

        self.assertEqual(free.stop_reason, "max_iterations")
        self.assertIn(patient.stop_reason, {"patience", "oscillation", "max_iterations"})
        # patience only truncates: same genes in the same order up to the stop
        self.assertEqual(patient.genes, free.genes[:len(patient.genes)])
        self.assertLessEqual(len(patient.flip_rates), len(patient.genes) - 1)




    def test_coexpression_null(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=3,
            seed=42,
            n_estimators=30,
            n_jobs=1,
            max_depth=5,
        )

        res = model.coexpression_null(n_shuffles=2)
        runs = res["runs"].set_index("run")
        self.assertEqual(runs.index.tolist(), ["observed", "shuffle_1", "shuffle_2"])
        self.assertEqual(runs.loc["observed", "jaccard"], 1.0)
        # shuffling genes across cells removes the planted group signal
        self.assertLess(runs.loc["shuffle_1", "first_auc"], runs.loc["observed", "first_auc"])
        self.assertTrue(set(res["flip_rates"]["run"]) <= set(runs.index))
        observed = res["panels"].query("run == 'observed'").sort_values("rank")["gene"]
        self.assertEqual(observed.tolist(), model.genes)
        # shuffled panels have the same length as the observed panel
        self.assertTrue((runs["n_genes"] == len(model.genes)).all())
        pv = res["pvalues"].set_index("statistic")
        self.assertTrue((pv["n_shuffles"] == 2).all())
        self.assertTrue(((pv["p_value"] >= 1 / 3) & (pv["p_value"] <= 1)).all())
        gf = res["gene_frequency"]
        self.assertEqual(gf["gene"].tolist(), model.genes)
        self.assertTrue(((gf["p_value"] >= 1 / 3) & (gf["p_value"] <= 1)).all())


    def test_null_pools_counts_and_recomputes_library_size(self):
        rng = np.random.default_rng(1)
        n = 2000
        # gene 0 low, gene 1 high and tied to labels and gene 2
        g1 = rng.poisson(np.repeat([2.0, 20.0], n // 2)).astype(np.float32)
        counts = np.column_stack([rng.poisson(1.0, n), g1, g1 + rng.poisson(1.0, n),
                                  rng.poisson(3.0, (n, 47))]).astype(np.float32)
        obs = pd.DataFrame({"group": np.repeat(["a", "b"], n // 2)})
        X, obs_s = recursieve.recursieve._pooled_null_data(
            counts, obs, np.random.default_rng(0))
        # counts keep their pooled values, and every gene follows the pool
        self.assertEqual(sorted(obs_s["group"]), sorted(obs["group"]))
        means = [np.expm1(X[:, j]).mean() for j in range(3)]
        self.assertLess(max(means) / min(means), 1.1)
        # library size is recomputed per cell from the shuffled counts
        pooled = np.random.default_rng(0).permuted(counts, axis=None)
        self.assertTrue(np.allclose(obs_s["log1p_total_counts"], np.log1p(pooled.sum(axis=1))))
        # genes, labels unrelated
        lab = (obs_s["group"] == "a").to_numpy().astype(float)
        for a, b in ((X[:, 1], X[:, 2]), (X[:, 1], lab)):
            self.assertLess(abs(np.corrcoef(a, b)[0, 1]), 0.1)


if __name__ == "__main__":
    unittest.main()
