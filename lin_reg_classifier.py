import anndata as ad    # type: ignore
import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack   # type: ignore
from sklearn.decomposition import IncrementalPCA    # type: ignore
import matplotlib.pyplot as plt

class LinRegClassifier:

    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = None


    def change_labels(self):
        labels = self.dataset.labels

        for i, label in enumerate(labels):
            if label != "normal":
                labels[i] = "diseased"

        print(f"Normal: {np.sum(labels == 'normal')/{len(labels)}}")
        print(f"Diseased: {np.sum(labels == 'diseased')}/{len(labels)}")

        self.labels = labels
        return labels

    def change_labels_to_numbers(labels):
        return np.where(labels == "normal", 0, 1).astype(int)

    