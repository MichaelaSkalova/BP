import anndata as ad    # type: ignore

class Dataset:

    def __init__(self, path):
        self.path = path
        self.adata = self.load_adata()

    def load_adata(self):
        adata = ad.read_h5ad(self.path, backed="r")
        return adata

    def keep_diseases(self):
        pass

    def get_percentile_subset(self):
        pass

    def train_test_split(self):
        pass

