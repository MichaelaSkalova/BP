import anndata as ad
import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

class LinRegClassifier:

    def __init__(self, dataset):
        self.dataset = dataset

    