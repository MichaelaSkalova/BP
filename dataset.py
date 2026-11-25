import anndata as ad    # type: ignore
import numpy as np
from analyzer import Analyzer

class Dataset():

    RANDOM_STATE = 42

    def __init__(self, path):
        self.path = path
        self.adata = self.load_adata()
        self.analysis = Analyzer(self)


    def load_adata(self):
        adata = ad.read_h5ad(self.path, backed="r")
        return adata


    def get_adata(self):
        return self.adata


    def keep_diseases(self, keep_diseases, sample_freq=1):
        all_indices = []
        rng = np.random.default_rng(self.RANDOM_STATE)

        for disease in keep_diseases:
            positions = np.flatnonzero(self.analysis.cell_diseases == disease)

            if positions.size == 0:
                continue

            if sample_freq == 1:
                keep_count = len(positions)
            else:
                keep_count = int(round(len(positions) * sample_freq))

            indices = np.sort(rng.choice(positions, size=keep_count, replace=False))
            all_indices.extend(indices.tolist())

        cell_indices = np.sort(np.asarray(all_indices, dtype=np.int64))

        print(
            f"Keeping {len(cell_indices):,}/"
            f"{self.adata.n_obs:,} cells."
        )

        new_path = str(self.path).replace(".h5ad", "_copy.h5ad")
        self.adata = self.adata[cell_indices].copy(new_path)
        self.path = new_path
        self.analysis = Analyzer(self)


    def get_variable_percentile_subset(self, percentile):
        pass


    def train_test_split(self):
        pass

