<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ArpiarSaundersLab/RecurSieve/main/manuscript/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ArpiarSaundersLab/RecurSieve/main/manuscript/logo_light.png">
    <img src="https://raw.githubusercontent.com/ArpiarSaundersLab/RecurSieve/main/manuscript/logo_light.png" alt="Recursieve logo" width="360">
  </picture>
</p>


**RecurSieve** is an iterative gene selection algorithm that identifies discriminative genes between two biological groups in single-cell datasets. It uses an ensemble learning approach with a blend of random forest classifiers and gaussian mixture models to progressively select genes that best separate the groups, collapsing selected genes into a lower dimensional representational axis at each iteration.

## Installation

Install from PyPI:

```bash
pip install recursieve
```

RecurSieve requires Python 3.12 or newer.

To install the development version from GitHub:

```bash
pip install git+https://github.com/ArpiarSaundersLab/RecurSieve.git
```

## Usage

```python
import anndata as ad
from recursieve import recursieve

adata = ad.read_h5ad("data/alz.h5ad")
model = recursieve(
    adata=adata,
    group1="control",
    group2="disease",
    field_name="condition",
    max_iterations=100,
)

# all genes found
panel = model.genes

# genes unique to RecurSieve
non_de_genes = model.unique_gene_panel 
 ```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| adata | AnnData | required | Annotated data matrix |
| group1 | str | required | First group label |
| group2 | str | required | Second group label |
| field_name | str | "sample" | Column in obs containing group labels |
| plots | bool | False | Generate visualization plots |
| print_to_console | bool | False | Print selected genes during iteration |
| max_iterations | int | 100 | Maximum iterations to run |
| additive | bool | True | Add new genes to panel or replace |
| flip_rate_percentage | float | 0.01 | Stop when the label flip rate falls below this value |
| patience | int or None | 10 | Stop after this many iterations without a new flip-rate minimum (None disables) |
| pval_cutoff | float | 0.05 | P-value threshold for differential expression |
| logfc_cutoff | float | 0.1 | Absolute log2 fold-change threshold for differential expression |
| seed | int | 42 | Random seed for reproducibility |
| summary_method | str | "mean" | Method to collapse genes: "mean" or "pca" |
| n_estimators | int | 300 | Number of random forest trees |
| n_jobs | int | -1 | Number of parallel jobs (-1 uses all cores) |
| max_depth | int or None | 10 | Maximum random forest tree depth |
| n_seeds | int | 10 | Seeds used by `seed_stability()` |
| preprocess | bool | True | Run QC, normalization, and HVG selection |
| run | bool | True | Run the pipeline on construction; if False, call `fit()` later |
| annsql_db | str or None | None | AnnSQL database to load instead of `adata` |

## Robustness checks

`seed_stability` checks whether the same genes are selected with different random seeds. `coexpression_null` checks whether the selected genes are found more often than by chance, by rerunning on randomized data.

Each seed and each shuffle is a full RecurSieve run, so these checks take much longer than a single run. On a 10-core laptop, 25 shuffles took about 1.5 hours for 25,000 cells and about 3.5 hours for 50,000 cells.

Both methods return a dictionary of pandas DataFrames and do not write files. The example below runs both checks and saves the main tables as CSV files.

```python
seeds = model.seed_stability(n_seeds=10)
null = model.coexpression_null(n_shuffles=25)

seeds["gene_frequency"].to_csv("seed_gene_frequency.csv", index=False)
null["pvalues"].to_csv("null_pvalues.csv", index=False)
null["gene_frequency"].to_csv("null_gene_frequency.csv", index=False)
```

<br>

## License

The RecurSieve software is released under the [MIT License](https://github.com/ArpiarSaundersLab/RecurSieve/blob/main/LICENSE). The data, figures, and tables in [`manuscript/`](https://github.com/ArpiarSaundersLab/RecurSieve/tree/main/manuscript) are dedicated to the public domain under [CC0 1.0](https://github.com/ArpiarSaundersLab/RecurSieve/blob/main/manuscript/LICENSE).

## Citation

RecurSieve: A Python Package Using Iterative Machine Learning to Detect Subtle Gene Expression Patterns in Single-Cell Data<br>
<i>bioRxiv link coming soon</i>