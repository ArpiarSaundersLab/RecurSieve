"""Load and subsample the Alzheimer's dataset.

All scripts load cells through load_alzheimers, so runs with the same options
use the same cells. The file is opened in backed mode and only the chosen rows
are read into memory.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

FIELD_NAME = "Disease.Group"
GROUPS = ("low", "high")


def add_loading_args(parser) -> None:
	"""Add the cell selection options."""
	parser.add_argument("--cell-type", default=None,
						help="Keep only this cell type (e.g. Excitatory)")
	parser.add_argument("--cell-type-key", default="Major_celltypes",
						help="obs column with cell types")
	parser.add_argument("--n-cells", type=int, default=50000,
						help="Cells to subsample")
	parser.add_argument("--balance-groups", action="store_true",
						help="Sample n-cells/2 low and n-cells/2 high cells")


def select_cells(obs, cell_type: str | None = None, cell_type_key: str = "Major_celltypes",
				 n_cells: int = 50000, balance_groups: bool = False, seed: int = 0) -> np.ndarray:
	"""Return row positions of the chosen cells, in output order."""
	pos = np.arange(len(obs))
	if cell_type is not None:
		pos = pos[(obs[cell_type_key] == cell_type).to_numpy()]
		print(f"Kept {cell_type} cells only: {len(pos)}")
	sub = obs.iloc[pos]

	if balance_groups:
		rng = np.random.default_rng(seed)
		keep = []
		for g in GROUPS:
			cells = np.flatnonzero((sub[FIELD_NAME] == g).to_numpy())
			keep.append(rng.choice(cells, size=min(n_cells // 2, len(cells)), replace=False))
			print(f"Sampled {len(keep[-1])} {g} cells")
		return pos[np.sort(np.concatenate(keep))]

	# same cells and order as sc.pp.subsample on the full data
	light = ad.AnnData(obs=sub.reset_index(drop=True).assign(_row=np.arange(len(sub))))
	light = sc.pp.subsample(light, n_obs=min(n_cells, len(sub)), random_state=seed, copy=True)
	return pos[light.obs["_row"].to_numpy()]


def load_alzheimers(adata_path: Path, cell_type: str | None = None,
					cell_type_key: str = "Major_celltypes", n_cells: int = 50000,
					balance_groups: bool = False, seed: int = 0) -> ad.AnnData:
	"""Load the chosen cells into memory with gene symbols as var_names."""
	print(f"Loading data from: {adata_path} (backed)")
	backed = ad.read_h5ad(adata_path, backed="r")
	order = select_cells(backed.obs, cell_type, cell_type_key, n_cells, balance_groups, seed)

	# backed indexing needs sorted rows
	rows = np.sort(order)
	adata = backed[rows].to_memory()
	backed.file.close()
	adata = adata[np.searchsorted(rows, order)].copy()

	adata.var_names = adata.var["feature_name"]
	adata.var_names_make_unique()
	return adata
