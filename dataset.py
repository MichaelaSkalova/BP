import anndata as ad    # type: ignore
import numpy as np
from analyzer import Analyzer


class Dataset():

    RANDOM_STATE = 42
    BATCH_SIZE = 512

    def __init__(self, path):
        self.path = path
        self.original_path = path
        self.adata = self.load_adata()
        self.analysis = Analyzer(self)


    def load_adata(self):
        adata = ad.read_h5ad(self.path, backed="r")
        return adata


    def get_adata(self):
        return self.adata


    def get_path(self):
        return self.path


    def get_original_path(self):
        return self.original_path


    def restore_original(self):
        self.path = self.original_path
        self.adata = self.load_adata()
        self.analysis = Analyzer(self)


    def read_batch(self, matrix, start, end):
        X_batch = matrix[start:end]
        return X_batch.tocsr(copy=False)


    def keep_diseases(self, keep_diseases):
        all_indices = []
        rng = np.random.default_rng(self.RANDOM_STATE)

        for disease in keep_diseases:
            positions = np.flatnonzero(self.analysis.cell_diseases == disease)

            if positions.size == 0:
                continue

            indices = np.sort(rng.choice(positions, size=len(positions), replace=False))
            all_indices.extend(indices.tolist())

        cell_indices = np.sort(np.asarray(all_indices, dtype=np.int64))

        print(
            f"Keeping {len(cell_indices):,}/"
            f"{self.adata.n_obs:,} cells."
        )

        new_path = str(self.path).replace(".h5ad", "_D.h5ad")
        self.adata = self.adata[cell_indices].copy(new_path)
        self.path = new_path
        self.analysis = Analyzer(self)


    def variable_percentile_subset(self, percentile):
        sums = np.zeros(self.adata.n_vars, dtype=np.float64)
        sum_squares = np.zeros(self.adata.n_vars, dtype=np.float64)
        n_cells = self.analysis.n_cells

        for start in range(0, n_cells, self.BATCH_SIZE):
            end = min(start + self.BATCH_SIZE, n_cells)

            X_batch = self.read_batch(self.adata.X, start, end)

            sums += np.asarray(X_batch.sum(axis=0)).ravel()
            sum_squares += np.asarray(X_batch.multiply(X_batch).sum(axis=0)).ravel()

        means = sums / n_cells
        mean_squares = sum_squares / n_cells

        variances = np.maximum(mean_squares - means**2, 0)
        cutoff = np.quantile(variances, 1 - percentile)
        var_mask = variances >= cutoff

        print(
            f"Keeping {var_mask.sum():,}/"
            f"{self.analysis.n_genes} genes."
        )

        new_path = str(self.path).replace(".h5ad", "_HV.h5ad")
        self.adata = self.adata[:, var_mask].copy(new_path)
        self.path = new_path
        self.analysis = Analyzer(self)


    def train_test_split(self):
        pass

