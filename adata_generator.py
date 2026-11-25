import numpy as np
import pandas as pd         # type: ignore
import anndata as ad        # type: ignore
from scipy import sparse    # type: ignore

np.random.seed(42)
OUTPUT_FILE = "synthetic_data_test.h5ad"

diseases = ["normal", "dementia", "schizophrenia", "alzheimer disease"]

n_cells = 1000
n_genes = 1200
cells_per_group = n_cells // len(diseases)

baseline_expression = 2.0
disease_effect = 1.5
noise_scale = 1.0

disease_labels = np.repeat(diseases, cells_per_group)
np.random.shuffle(disease_labels)

cell_types = ["neuron", "astrocyte", "microglia", "oligodendrocyte"]
cell_type_labels = np.random.choice(
    cell_types,
    size=n_cells,
    p=[0.45, 0.25, 0.15, 0.15]
)

donor_id = []

for disease in diseases:
    for donor in range(5):
        donor_id.extend(
            [f"{disease}_Donor_{donor+1}"] * 50
        )

donor_id = np.array(donor_id)   # type: ignore

metadata = pd.DataFrame({
    "disease": np.repeat(diseases, 250),
    "donor_id": donor_id
})

metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

cell_type_labels = np.random.choice(
    cell_types,
    size=n_cells,
    p=[0.45, 0.25, 0.15, 0.15]
)

obs = pd.DataFrame({
    "disease": metadata["disease"].values,
    "donor_id": metadata["donor_id"].values,
    "cell_type": cell_type_labels,
    "sample_id": [
        f"Sample_{i // 50 + 1}" for i in range(n_cells)
    ]
}, index=[f"Cell_{i}" for i in range(n_cells)])

gene_categories = {
    "housekeeping": 250,
    "neuronal": 200,
    "inflammatory": 200,
    "immune": 150,
    "disease_associated": 200,
    "metabolic": 100,
    "random": 100
}

gene_types = []
for category, number in gene_categories.items():
    gene_types.extend([category] * number)

gene_types = np.array(gene_types)   # type: ignore
np.random.shuffle(gene_types)

var = pd.DataFrame({
    "feature_type": gene_types,
}, index=[f"Gene_{i}" for i in range(n_genes)])


base_expression = np.random.gamma(
    shape=2.0,
    scale=1.5,
    size=n_genes
)

high_expression_genes = np.random.choice(n_genes, 100, replace=False)
base_expression[high_expression_genes] *= 4

expression = np.zeros((n_cells, n_genes))

for i in range(n_cells):
    disease = obs.iloc[i]["disease"]
    cell_type = obs.iloc[i]["cell_type"]

    gene_means = base_expression.copy()

    if disease == "normal":
        pass

    elif disease == "dementia":
        gene_means[gene_types == "neuronal"] *= 0.75
        gene_means[gene_types == "inflammatory"] *= 1.8
        gene_means[gene_types == "immune"] *= 1.4
        gene_means[gene_types == "disease_associated"] *= 1.4

    elif disease == "schizophrenia":
        gene_means[gene_types == "neuronal"] *= 1.5
        gene_means[gene_types == "metabolic"] *= 0.8
        gene_means[gene_types == "immune"] *= 1.2
        gene_means[gene_types == "disease_associated"] *= 1.3

    elif disease == "alzheimer disease":
        gene_means[gene_types == "neuronal"] *= 0.6
        gene_means[gene_types == "inflammatory"] *= 2.5
        gene_means[gene_types == "immune"] *= 2.0
        gene_means[gene_types == "disease_associated"] *= 2.5

    if cell_type == "neuron":
        gene_means[gene_types == "neuronal"] *= 2.0

    elif cell_type == "astrocyte":
        gene_means[gene_types == "metabolic"] *= 1.5

    elif cell_type == "microglia":
        gene_means[gene_types == "immune"] *= 2.0
        gene_means[gene_types == "inflammatory"] *= 1.5

    elif cell_type == "oligodenderocyte":
        gene_means[gene_types == "metabolic"] *= 1.5


    cell_library_factor = np.random.lognormal(mean=0, sigma=0.3)
    gene_means *= cell_library_factor

    dispersion = 2.0
    p = dispersion / (dispersion + gene_means)
    counts = np.random.negative_binomial(n=dispersion, p=p)

    dropout_probability = 0.05
    dropout_mask = np.random.random(n_genes) < dropout_probability
    counts[dropout_mask] = 0
    expression[i, :] = counts


adata = ad.AnnData(
    X=sparse.csr_matrix(expression),
    obs=obs,
    var=var
)

adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()
adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
adata.var["total_counts"] = np.asarray(adata.X.sum(axis=0)).flatten()
adata.var["n_cells_by_counts"] = np.asarray((adata.X > 0).sum(axis=0)).flatten()

gene_variance = np.asarray(adata.X.toarray().var(axis=0)).flatten()

adata.write_h5ad(OUTPUT_FILE)

print(adata)
print("\nDisease distribution: ")
print(adata.obs["disease"].value_counts())
print("\nGene type distribution: ")
print(adata.var["feature_type"].value_counts())