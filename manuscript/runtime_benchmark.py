"""Runtime of recursieve on simulated data of increasing size.

Runs recursieve 3 times on simulated datasets of 2000 genes and 5k to 25k
cells. Every run does exactly N_ITERATIONS iterations (stopping rules off):
with the defaults, the number of iterations depends on the seed and varies
more than tenfold, which hides how runtime scales with dataset size. Existing
results in the output folder are resumed; runs made with a different
N_ITERATIONS are ignored.

Usage:
    python runtime_benchmark.py [--out-dir DIR]

Outputs (to DIR):
- runtime_benchmark_results.csv: Runtime of every run
- runtime_benchmark_summary.csv: Mean, SD, median, min, and max per size
- runtime_benchmark_barplot.png / .pdf: median runtime, min-max range, and
  each run as a dot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import recursieve
from scsim import scsim


GENE_COUNT = 2000
CELL_COUNTS = [5000, 10000, 15000, 20000, 25000]
N_REPEATS = 3
N_ITERATIONS = 20  # iterations per run, fixed so runs are comparable


def build_dataset(ncells: int, ngenes: int, seed: int) -> ad.AnnData:
    """Simulate one dataset with group_1 and group_2 labels."""
    simulator = scsim(
        ngenes=ngenes,
        ncells=ncells,
        ngroups=2,
        nproggenes=0,
        seed=seed,
    )
    simulator.simulate()

    adata = ad.AnnData(
        X=simulator.counts.values,
        obs=simulator.cellparams.copy(),
        var=simulator.geneparams.copy(),
    )
    adata.obs_names = simulator.counts.index.astype(str)
    adata.var_names = simulator.counts.columns.astype(str)
    adata.obs["group"] = adata.obs["group"].map({1: "group_1", 2: "group_2"}).astype("category")
    return adata


def time_recursieve(adata: ad.AnnData, seed: int) -> tuple[float, int]:
    """Run recursieve once for N_ITERATIONS iterations.

    Returns runtime in seconds and the number of genes selected.
    """
    start = time.perf_counter()
    model = recursieve.recursieve(
        adata.copy(),
        group1="group_1",
        group2="group_2",
        field_name="group",
        plots=False,
        print_to_console=False,
        n_jobs=-1,
        seed=seed,
        # stopping rules off: a negative threshold disables the threshold and
        # oscillation stops, and patience=None disables the patience stop
        max_iterations=N_ITERATIONS,
        flip_rate_percentage=-1.0,
        patience=None,
    )
    return time.perf_counter() - start, len(model.genes)


def plot_results(summary_df: pd.DataFrame, results_df: pd.DataFrame,
                 out_png: Path, out_pdf: Path) -> None:
    """Bar plot of median runtime with min-max range and every run as a dot."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = summary_df.sort_values("cells").reset_index(drop=True)
    labels = [f"{x // 1000}k" for x in plot_df["cells"]]
    x = range(len(plot_df))
    median = plot_df["median_runtime_sec"]
    # min-max range around the median (never below zero, unlike mean +/- SD)
    yerr = [median - plot_df["min_runtime_sec"], plot_df["max_runtime_sec"] - median]
    ax.bar(x, median, yerr=yerr, capsize=6, color="#94b6d8", edgecolor="black",
           linewidth=1.0, error_kw={"elinewidth": 1.2})
    for i, n_cells in enumerate(plot_df["cells"]):
        runs = results_df.loc[results_df["cells"] == n_cells, "runtime_sec"]
        ax.scatter([i] * len(runs), runs, s=25, color="#1a1d21", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"recursieve Runtime vs Dataset Size ({N_ITERATIONS} iterations)",
                 pad=12, weight="bold")
    ax.set_xlabel("Number of Cells (2000 genes)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # median label just above the top of the range
    pad = 0.02 * plot_df["max_runtime_sec"].max()
    for i, (val, top) in enumerate(zip(median, plot_df["max_runtime_sec"])):
        ax.text(i, top + pad, f"{val:.1f}s", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "robustness" / "runtime_benchmark")
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "runtime_benchmark_results.csv"
    summary_path = out_dir / "runtime_benchmark_summary.csv"
    plot_png = out_dir / "runtime_benchmark_barplot.png"
    plot_pdf = out_dir / "runtime_benchmark_barplot.pdf"

    rows: list[dict] = []
    done = set()

    if results_path.exists():
        existing = pd.read_csv(results_path)
        # only runs made with this iteration count can be combined
        if "iterations" in existing.columns:
            existing = existing[existing["iterations"] == N_ITERATIONS]
        else:
            existing = existing.iloc[0:0]
        rows = existing.to_dict("records")
        done = {(int(r["cells"]), int(r["replicate"])) for r in rows}
        print(f"Resuming from existing results: {len(rows)} completed runs "
              f"with {N_ITERATIONS} iterations")

    for i, n_cells in enumerate(CELL_COUNTS):
        dataset_seed = 1000 + i
        print(f"\nGenerating dataset: {n_cells} cells x {GENE_COUNT} genes (seed={dataset_seed})")
        adata = build_dataset(ncells=n_cells, ngenes=GENE_COUNT, seed=dataset_seed)

        for rep in range(1, N_REPEATS + 1):
            if (n_cells, rep) in done:
                print(f"  Skipping replicate {rep}/{N_REPEATS}; already completed")
                continue
            run_seed = 2000 + i * 10 + rep
            print(f"  Running recursieve replicate {rep}/{N_REPEATS} (seed={run_seed})")
            runtime_sec, n_selected = time_recursieve(adata=adata, seed=run_seed)
            print(f"    Runtime: {runtime_sec:.3f} sec ({n_selected} genes selected)")
            rows.append(
                {
                    "cells": n_cells,
                    "genes": GENE_COUNT,
                    "replicate": rep,
                    "dataset_seed": dataset_seed,
                    "run_seed": run_seed,
                    "iterations": N_ITERATIONS,
                    "genes_selected": n_selected,
                    "runtime_sec": runtime_sec,
                }
            )
            pd.DataFrame(rows).to_csv(results_path, index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["cells", "genes"], as_index=False)["runtime_sec"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_runtime_sec",
                "std": "std_runtime_sec",
                "median": "median_runtime_sec",
                "min": "min_runtime_sec",
                "max": "max_runtime_sec",
            }
        )
    )
    summary_df.to_csv(summary_path, index=False)

    plot_results(summary_df, results_df, out_png=plot_png, out_pdf=plot_pdf)

    print(summary_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
