import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score
from pathlib import Path
from alz_loading import load_alzheimers

#fields
FIELD_NAME = "Disease.Group"
GROUPS = ("low", "high")
DATA_PATH = "/home/kenny/Documents/OHSU/Projects/Anushi Algo/data/alz.h5ad"

#figure settings
FIGURE_DPI = 300
FIGURE_FONT_SETTINGS = {"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42}
GROUP_COLORS = {"low": "#0072B2", "high": "#E69F00"}
PANEL_FONT_SIZE = 7
#white background (even with a dark editor theme), arial, and small text for both panels
PANEL_STYLE = ["default", FIGURE_FONT_SETTINGS,
               {"font.size": PANEL_FONT_SIZE, "axes.titlesize": PANEL_FONT_SIZE,
                "axes.labelsize": PANEL_FONT_SIZE, "xtick.labelsize": PANEL_FONT_SIZE,
                "ytick.labelsize": PANEL_FONT_SIZE}]

#distribution panel: every gene in two columns, size is width x height in inches
PANEL_E_PATH = "distribution_assessment/panel_E"
PANEL_E_SIZE = (3.5, 4.5)
#y axis top = tallest bar x this, leaves room for the mean text
PANEL_E_HEADROOM = 1.3
#x axis end for every distribution, the few cells above this are off the plot
PANEL_E_X_MAX = 2.5

#scatter panel: KS vs wasserstein percentile, one scatter per run side by side
SCATTER_PANEL_PATH = "distribution_assessment/panel_scatter"
SCATTER_PANEL_SIZE = (3.5, 2.2)
#labels are right-aligned to this x (fraction of the axes width) so every connector starts past all the text
SCATTER_LABEL_END = 0.45
#gene labels in the scatters are a bit smaller than the rest of the text so they fit
SCATTER_LABEL_FONT_SIZE = 6

#table with the KS and wasserstein percentiles and distribution stats for every gene
SUMMARY_PATH = "distribution_assessment/distribution_summary.csv"

#settings for each recursieve run in the paper, and the genes reported from that run
#genes = labeled in the scatter and shown in panel E
#recursieve_genes = every gene recursieve found in that run (Supplemental Table 1), black points in the scatter
RUNS = {
    "all_cell_types": {
        "title": "All cell types",
        "cell_type": None, "n_cells": 50000, "balance_groups": False,
        "genes": ["WWOX", "CNTN5", "HSP90AA1", "KCNIP4", "HS6ST3", "DPP10"],
        "recursieve_genes": ["WWOX", "NCAM2", "HSP90AA1", "ERBB4", "CNTN5", "DPP10", "HS6ST3", "PDE4B",
                             "KCNIP4", "SNHG14", "DOCK4", "SPOCK1", "RALYL", "LUC7L3", "RBM25", "QKI",
                             "ZEB2", "PLXDC2", "COPG2", "SLC24A2", "ANKRD12", "FRMD5", "FHIT", "ROBO2",
                             "PLCL1", "ELMO1", "NPAS3", "SLC44A1", "ARAP2", "MAN2A1", "DPYD", "GRID2",
                             "DIP2B", "FRYL", "TCF12", "ZC3H13", "EDIL3", "ZNF91", "ENSG00000230021",
                             "PIP4K2A", "SLAIN1"],
    },
    "excitatory": {
        "title": "Excitatory neurons",
        "cell_type": "Excitatory", "n_cells": 25000, "balance_groups": True,
        "genes": ["LRP1B", "CNTNAP2"],
        "recursieve_genes": ["HSP90AA1", "MAP1B", "AKAP9", "LRP1B", "DPP10", "CNTNAP2", "SNHG14", "LUC7L3",
                             "RBM25", "SRSF11", "LRFN5", "COPG2", "TENM2", "PNISR", "CDH12", "ANKRD26",
                             "ANKRD12", "PRRC2C", "ENSG00000274422", "BDP1", "TAFA2", "EPIC1", "ZC3H13"],
    },
}


def load_and_preprocess(settings):
    #load the same cells the recursieve run used (gene names are made unique here too)
    adata = load_alzheimers(DATA_PATH, cell_type=settings["cell_type"], n_cells=settings["n_cells"],
                            balance_groups=settings["balance_groups"])

    #same preprocessing as recursieve.preprocessing, so the top 2000 genes match
    #drop mitochondrial, ribosomal, and hemoglobin genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.match(r"^RPS|^RPL")
    adata.var["hb"] = adata.var_names.str.upper().str.startswith("HB")
    adata = adata[:, ~(adata.var["mt"] | adata.var["ribo"] | adata.var["hb"])].copy()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=10)
    #raw counts -> log-normalized expression
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def collect_gene_values(adata, genes):
    #each gene's expression split by group, zeros included, kept for the figure after adata is freed
    gene_values = {}
    for gene in genes:
        position = np.where(adata.var_names == gene)[0][0]
        values = adata.X[:, position].toarray().ravel()
        gene_values[gene] = {group: values[(adata.obs[FIELD_NAME] == group).to_numpy()] for group in GROUPS}
    return gene_values


def score_genes(adata, genes):
    #calculate most variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    #pull out the top 2000 gene columns once, by position
    top_positions = np.where(adata.var["highly_variable"])[0]
    top_genes = set(adata.var_names[top_positions])

    #also score the genes of interest that didn't make the top 2000
    gene_positions = [np.where(adata.var_names == gene)[0][0] for gene in genes]
    extra_positions = [position for position in gene_positions if position not in top_positions]
    score_positions = np.concatenate([top_positions, extra_positions]).astype(int)
    score_names = adata.var_names[score_positions]
    score_matrix = adata.X[:, score_positions]

    #split cells into the two groups once
    #column format (csc) makes grabbing one gene at a time fast
    low_matrix = score_matrix[(adata.obs[FIELD_NAME] == GROUPS[0]).to_numpy()].tocsc()
    high_matrix = score_matrix[(adata.obs[FIELD_NAME] == GROUPS[1]).to_numpy()].tocsc()

    #wasserstein distance and KS (Kolmogorov-Smirnov) statistic for each gene
    #KS = biggest gap between the two cumulative distributions, 0 to 1, so it doesn't depend on expression level
    wasserstein_results = {}
    ks_results = {}
    for column, gene in enumerate(score_names):
        low_values = low_matrix[:, column].toarray().ravel()
        high_values = high_matrix[:, column].toarray().ravel()
        wasserstein_results[gene] = wasserstein_distance(low_values, high_values)
        #use the statistic, not the p-value, since with this many cells nearly every p-value is tiny
        ks_results[gene] = ks_2samp(low_values, high_values).statistic

    return pd.Series(wasserstein_results), pd.Series(ks_results), top_genes


def print_summary(genes, wasserstein_results, ks_results, top_genes):
    #both measures for the genes of interest, plus percentile within the top 2000 (100 = most different)
    top_wasserstein = wasserstein_results[wasserstein_results.index.isin(top_genes)]
    top_ks = ks_results[ks_results.index.isin(top_genes)]
    for gene in genes:
        wasserstein_percentile = (top_wasserstein < wasserstein_results[gene]).mean() * 100
        ks_percentile = (top_ks < ks_results[gene]).mean() * 100
        note = "" if gene in top_genes else " (not in top 2000)"
        print(f"{gene}{note}: wasserstein = {wasserstein_results[gene]:.3f} ({wasserstein_percentile:.0f}th percentile), "
              f"KS = {ks_results[gene]:.3f} ({ks_percentile:.0f}th percentile)")


def plot_percentile_scatter(ax, wasserstein_results, ks_results, genes, recursieve_genes, top_genes, title,
                            show_ylabel):
    #each top 2000 gene placed by its wasserstein percentile (x) and KS percentile (y), 100 = most different
    #percentile = share of the top 2000 genes with a smaller value, same as print_summary
    top_wasserstein = wasserstein_results[wasserstein_results.index.isin(top_genes)]
    top_ks = ks_results[ks_results.index.isin(top_genes)]
    n_genes = len(top_wasserstein)
    wasserstein_percentile = (top_wasserstein.rank(method="min") - 1) / n_genes * 100
    ks_percentile = (top_ks.rank(method="min") - 1) / n_genes * 100
    #rasterized keeps the pdf small with 2000 points, the text and axes stay vector
    ax.scatter(wasserstein_percentile, ks_percentile[wasserstein_percentile.index], s=1.5, color="gray",
               alpha=0.4, linewidths=0, rasterized=True)

    #dashed lines mark the top 10% on each measure
    ax.axvline(90, color="lightgray", linestyle="--", linewidth=0.6, zorder=0)
    ax.axhline(90, color="lightgray", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    #square plot, bold header to match the distribution panel
    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=PANEL_FONT_SIZE + 1, fontweight="bold")
    ax.set_xlabel("Wasserstein percentile")
    ax.set_ylabel("KS percentile" if show_ylabel else "")
    ax.spines[["top", "right"]].set_visible(False)

    #every gene recursieve found in black, no label
    for gene in recursieve_genes:
        gene_wasserstein = (top_wasserstein < wasserstein_results[gene]).mean() * 100
        gene_ks = (top_ks < ks_results[gene]).mean() * 100
        ax.scatter(gene_wasserstein, gene_ks, color="black", s=2, zorder=3, clip_on=False)

    #genes of interest in black, labels stacked on the left and right-aligned, highest KS on top
    gene_percentiles = []
    for gene in genes:
        gene_wasserstein = (top_wasserstein < wasserstein_results[gene]).mean() * 100
        gene_ks = (top_ks < ks_results[gene]).mean() * 100
        gene_percentiles.append((gene, gene_wasserstein, gene_ks))
    gene_percentiles.sort(key=lambda row: row[2], reverse=True)

    for step, (gene, gene_wasserstein, gene_ks) in enumerate(gene_percentiles):
        #clip_on=False so a point at the very top or right edge isn't cut in half
        ax.scatter(gene_wasserstein, gene_ks, color="black", s=2, zorder=3, clip_on=False)
        #start below the dashed 90th percentile line so it doesn't run through the labels
        label_height = 0.84 - step * 0.1
        #gene name only, the exact percentiles go in the caption or supplemental table
        ax.annotate(gene, xy=(gene_wasserstein, gene_ks), fontsize=SCATTER_LABEL_FONT_SIZE,
                    xytext=(SCATTER_LABEL_END, label_height), textcoords="axes fraction", ha="right", va="center",
                    #faint white box so labels stay readable over the gray points
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
                    #relpos (1, 0.5) starts the connector at the right edge of the label, not its middle
                    arrowprops={"arrowstyle": "-", "color": "gray", "linewidth": 0.5, "relpos": (1, 0.5),
                                "shrinkA": 2, "shrinkB": 2})


def plot_scatter_panel(run_results):
    #one KS vs wasserstein percentile scatter per run, side by side
    with plt.style.context(PANEL_STYLE):
        fig, axes = plt.subplots(1, len(RUNS), figsize=SCATTER_PANEL_SIZE, layout="constrained", squeeze=False)
        for column, (ax, (run_name, settings)) in enumerate(zip(axes.flat, RUNS.items())):
            wasserstein_results, ks_results, top_genes, gene_values = run_results[run_name]
            plot_percentile_scatter(ax, wasserstein_results, ks_results, settings["genes"],
                                    settings["recursieve_genes"], top_genes, settings["title"],
                                    show_ylabel=(column == 0))

        Path(SCATTER_PANEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{SCATTER_PANEL_PATH}.png", dpi=FIGURE_DPI)
        fig.savefig(f"{SCATTER_PANEL_PATH}.pdf")
        plt.show()


def ordinal(number):
    #93 -> "93rd", 100 -> "100th", 11 -> "11th"
    number = int(round(number))
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def plot_gene_panel(ax, gene, values_by_group, x_max):
    #low and high overlaid for one gene (non-zero cells), with the mean shift centered under the gene name
    bins = np.linspace(0, max(values.max() for values in values_by_group.values()), 50)
    group_means = {}
    for group in GROUPS:
        values = values_by_group[group]
        nonzero = values[values > 0]
        group_means[group] = nonzero.mean()
        #density so group size doesn't matter
        sns.histplot(nonzero, bins=bins, stat="density", element="step", alpha=0.3,
                     linewidth=0.8, color=GROUP_COLORS[group], ax=ax)
    ax.set_title(gene, pad=2)

    #mean shift = high mean - low mean (non-zero cells), in the empty space above the distributions
    mean_shift = group_means[GROUPS[1]] - group_means[GROUPS[0]]
    ax.set_ylim(0, ax.get_ylim()[1] * PANEL_E_HEADROOM)
    ax.text(0.5, 0.98, f"Mean shift: {mean_shift:+.2f}", transform=ax.transAxes, ha="center", va="top",
            fontsize=PANEL_FONT_SIZE - 1, color="dimgray")

    #same x range in every panel so the genes can be compared, and no y axis since only the shapes matter
    ax.set_xlim(0, x_max)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_panel_e(run_results):
    #compact version for the main figure: one block per analysis with a header, genes in two columns
    rows_per_run = [int(np.ceil(len(settings["genes"]) / 2)) for settings in RUNS.values()]
    #shared x range for every gene, set in the settings
    x_max = PANEL_E_X_MAX

    with plt.style.context(PANEL_STYLE):
        fig = plt.figure(figsize=PANEL_E_SIZE, layout="constrained")
        #each block's height = its rows plus a bit for its header, so gene panels end up the same height
        subfigs = np.atleast_1d(fig.subfigures(len(RUNS), 1, height_ratios=[rows + 0.4 for rows in rows_per_run]))
        for subfig, n_rows, (run_name, settings) in zip(subfigs, rows_per_run, RUNS.items()):
            gene_values = run_results[run_name][3]
            axes = subfig.subplots(n_rows, 2, sharex=True, squeeze=False)
            for ax, gene in zip(axes.flat, settings["genes"]):
                plot_gene_panel(ax, gene, gene_values[gene], x_max)
            #hide the leftover slot if there's an odd number of genes
            for ax in axes.flat[len(settings["genes"]):]:
                ax.axis("off")
            #header for the analysis, so it isn't repeated on every gene
            subfig.suptitle(settings["title"], fontsize=PANEL_FONT_SIZE + 1, fontweight="bold")

        #one shared legend and x label instead of one per panel
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[group], alpha=0.5) for group in GROUPS]
        fig.legend(handles, [f"{group} pathology" for group in GROUPS], loc="outside upper center", ncol=2,
                   frameon=False)
        fig.supxlabel("Expression (log-normalized, non-zero cells)", fontsize=PANEL_FONT_SIZE)

        Path(PANEL_E_PATH).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{PANEL_E_PATH}.png", dpi=FIGURE_DPI)
        fig.savefig(f"{PANEL_E_PATH}.pdf")
        plt.show()


def write_summary(run_results):
    #one row per gene: KS and wasserstein with percentiles, plus the main stats of each group's distribution
    rows = []
    for run_name, settings in RUNS.items():
        wasserstein_results, ks_results, top_genes, gene_values = run_results[run_name]
        top_wasserstein = wasserstein_results[wasserstein_results.index.isin(top_genes)]
        top_ks = ks_results[ks_results.index.isin(top_genes)]
        for gene in settings["genes"]:
            row = {
                "analysis": settings["title"],
                "gene": gene,
                "in_top_2000": gene in top_genes,
                "ks_statistic": ks_results[gene],
                #percentile = share of the top 2000 genes with a smaller value (100 = most different)
                "ks_percentile": (top_ks < ks_results[gene]).mean() * 100,
                "wasserstein_distance": wasserstein_results[gene],
                "wasserstein_percentile": (top_wasserstein < wasserstein_results[gene]).mean() * 100,
            }
            group_stats = {}
            for group in GROUPS:
                values = gene_values[gene][group]
                nonzero = values[values > 0]
                q25, median, q75 = np.percentile(nonzero, [25, 50, 75])
                group_stats[group] = {
                    "n_cells": len(values),
                    "zero_percent": (values == 0).mean() * 100,
                    #mean over all cells (zeros included), closest to what standard DE compares
                    "mean_all_cells": values.mean(),
                    #the rest describe the expressing (non-zero) cells, same as the distribution plots
                    "mean_nonzero": nonzero.mean(),
                    "median_nonzero": median,
                    "sd_nonzero": nonzero.std(),
                    "iqr_nonzero": q75 - q25,
                }
                for stat, value in group_stats[group].items():
                    row[f"{stat}_{group}"] = value

            #high minus low for the main stats, positive = higher in high pathology
            low, high = group_stats[GROUPS[0]], group_stats[GROUPS[1]]
            row["zero_percent_change"] = high["zero_percent"] - low["zero_percent"]
            row["mean_shift_all_cells"] = high["mean_all_cells"] - low["mean_all_cells"]
            row["mean_shift_nonzero"] = high["mean_nonzero"] - low["mean_nonzero"]
            row["sd_change_nonzero"] = high["sd_nonzero"] - low["sd_nonzero"]
            row["iqr_change_nonzero"] = high["iqr_nonzero"] - low["iqr_nonzero"]
            rows.append(row)

    summary = pd.DataFrame(rows)
    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(SUMMARY_PATH, index=False)
    print(f"Saved summary to {SUMMARY_PATH}")
    return summary


#run each recursieve setup: score its top 2000 genes and keep its genes' values for the figure
run_results = {}
for run_name, settings in RUNS.items():
    print(f"\n{settings['title']}")
    adata = load_and_preprocess(settings)
    #score the labeled genes plus every recursieve gene, so all of them can go on the scatter
    score_list = list(dict.fromkeys(settings["genes"] + settings["recursieve_genes"]))
    wasserstein_results, ks_results, top_genes = score_genes(adata, score_list)
    print_summary(settings["genes"], wasserstein_results, ks_results, top_genes)
    gene_values = collect_gene_values(adata, settings["genes"])
    run_results[run_name] = (wasserstein_results, ks_results, top_genes, gene_values)
    #free memory before loading the next run
    del adata

#gene distributions in two columns, then the KS vs wasserstein scatters
plot_panel_e(run_results)
plot_scatter_panel(run_results)

#summary table of every gene's percentiles and distribution stats
summary = write_summary(run_results)
