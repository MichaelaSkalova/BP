import anndata as ad    # type: ignore
import numpy as np
from analyzer import Analyzer
from scipy.sparse import issparse, csr_matrix, hstack   # type: ignore


class Dataset():

    RANDOM_STATE = 42
    BATCH_SIZE = 512

    def __init__(self, path):
        self.path = path
        self.original_path = path
        self.adata = self.load_adata()
        self.analysis = Analyzer(self)

        self.data = None
        self.labels = None


    def load_adata(self):
        adata = ad.read_h5ad(self.path, backed="r")
        return adata


    def restore_original(self):
        self.path = self.original_path
        self.adata = self.load_adata()
        self.analysis = Analyzer(self)


    def read_batch(self, matrix, start, end):
        X_batch = matrix[start:end]
        return X_batch.tocsr(copy=False)


    def keep_diseases(self, keep_diseases):
        all_indices = []

        for disease in keep_diseases:
            positions = np.flatnonzero(self.analysis.cell_diseases == disease)

            if positions.size == 0:
                continue

            all_indices.extend(positions.tolist())

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


    def get_data_and_labels(self):
        data = csr_matrix(self.adata.X[:], dtype=np.float32)

        data.indices = data.indices.astype(np.int32)
        data.indptr = data.indptr.astype(np.int32)

        labels_dict = dict()
        labels = list()

        for i in range(self.adata.n_obs):
            label = self.adata.obs["diseaes"].iloc[i]

            if label not in labels_dict.keys():
                labels_dict[label] = list()

            labels_dict[label].append(i)
            labels.append(label)

        self.csr_data = data
        self.labels_dict = labels_dict
        self.labels = np.asarray(labels)

        return data, labels_dict, labels


    def get_part(self, percent):
        rng = np.random.default_rng(self.RANDOM_STATE)
        test_inds_all = []

        for key, inds in self.labels_dict.items():
            inds = np.asarray(inds, dtype=np.int32)
            test_len = int(len(inds) * percent)
            test_inds = rng.choice(inds, size=test_len, replace=False)
            test_inds_all.extend(test_inds.tolist())

        test_inds_all = np.asarray(sorted(test_inds_all), dtype=np.int32)
        train_inds_all = np.setdiff1d(np.arange(self.data.shape[0], dtype=np.int32), test_inds_all)

        self.train_data = self.data[train_inds_all, :]
        data1 = self.data[train_inds_all, :]

        self.test_data = self.data[test_inds_all, :]
        data2 = self.data[test_inds_all, :]

        self.train_labels = self.labels[train_inds_all]
        labels1 = self.labels[train_inds_all]

        self.test_labels = self.labels[test_inds_all]
        labels2 = self.labels[test_inds_all]

        return data1, data2, labels1, labels2

