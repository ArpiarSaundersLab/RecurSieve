"""Runtime of recursieve on simulated data of increasing size.

Runs recursieve 3 times on simulated datasets of 2000 genes and 5k to 25k
cells. Existing results in the output folder are resumed.

Usage:
    python runtime_benchmark.py [--out-dir DIR]

Outputs (to DIR):
- runtime_benchmark_results.csv: Runtime of every run
- runtime_benchmark_summary.csv: Mean, SD, min, and max per size
- runtime_benchmark_barplot.png / .pdf
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


def time_recursieve(adata: ad.AnnData, seed: int) -> float:
    """Run recursieve once and return runtime in seconds."""
    start = time.perf_counter()
    _ = recursieve.recursieve(
        adata.copy(),
        group1="group_1",
        group2="group_2",
        field_name="group",
        plots=False,
        print_to_console=False,
        n_jobs=-1,
        seed=seed,
    )
    return time.perf_counter() - start


def plot_results(summary_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    """Bar plot of mean runtime with SD error bars."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = summary_df.sort_values("cells").copy()
    plot_df["cells_label"] = plot_df["cells"].map(lambda x: f"{x // 1000}k")

    palette = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"]
    ax.bar(
        plot_df["cells_label"],
        plot_df["mean_runtime_sec"],
        yerr=plot_df["std_runtime_sec"],
        capsize=6,
        color=palette,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title("recursieve Runtime vs Dataset Size", pad=12, weight="bold")
    ax.set_xlabel("Number of Cells (2000 genes)")
    ax.set_ylabel("Runtime (seconds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, val in enumerate(plot_df["mean_runtime_sec"]):
        ax.text(i, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=10)

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
        rows = existing.to_dict("records")
        done = {(int(r["cells"]), int(r["replicate"])) for r in rows}
        print(f"Resuming from existing results: {len(rows)} completed runs")

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
            runtime_sec = time_recursieve(adata=adata, seed=run_seed)
            print(f"    Runtime: {runtime_sec:.3f} sec")
            rows.append(
                {
                    "cells": n_cells,
                    "genes": GENE_COUNT,
                    "replicate": rep,
                    "dataset_seed": dataset_seed,
                    "run_seed": run_seed,
                    "runtime_sec": runtime_sec,
                }
            )
            pd.DataFrame(rows).to_csv(results_path, index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["cells", "genes"], as_index=False)["runtime_sec"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_runtime_sec",
                "std": "std_runtime_sec",
                "min": "min_runtime_sec",
                "max": "max_runtime_sec",
            }
        )
    )
    summary_df.to_csv(summary_path, index=False)

    plot_results(summary_df, out_png=plot_png, out_pdf=plot_pdf)

    print(summary_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()