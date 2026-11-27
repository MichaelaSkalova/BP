import numpy as np
import pandas as pd     # type: ignore

class Analyzer():

    def __init__(self, dataset):

        self._adata = dataset.adata
        self.n_cells, self.n_genes = self.get_n()
        self.diseases = self.get_diseases()
        self.cell_diseases = self.get_cell_diseases()

        self.gene_types = self.get_gene_types()
        self.donor_ids = self.get_donor_ids()
        self.donor_amount = len(self.donor_ids)

    #=====================================
    # FUNCTIONS FOR CALCULATING ATTRIBUTES

    def get_n(self) -> tuple[int, int]:
        "returns number of cells, number of genes"
        n_obs = self._adata.n_obs
        n_vars = self._adata.n_vars
        return n_obs, n_vars


    def get_diseases(self) -> list[str]:
        "returns list of unique diseases"
        data = self._adata.obs["disease"]
        unique_diseases = data.dropna().unique().tolist()
        return unique_diseases


    def get_cell_diseases(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        "returns array of the disease for each cell"
        return self._adata.obs["disease"].to_numpy()


    def get_gene_types(self) -> pd.Series:
        '''returns pd.Series of gene types and amounts of genes per type.
        - individual values can be accessed:
        gene_types.index -> feature types, gene_types.values -> amount, gene_types[type] -> amount for type'''
        gene_types = self._adata.var["feature_type"].value_counts(dropna=False)

        return gene_types


    def get_donor_ids(self) -> list[str]:
        "returns list of unique donor ids"
        data = self._adata.obs["donor_id"]
        unique_ids = data.dropna().unique().tolist()

        return unique_ids

    # ====================================
    # FUNCTIONS FOR WHOLE DATASET ANALYSIS

    def get_cells_per_disease(self) -> tuple[dict[str, int], dict[str, int]]:
        "returns dicts of {'disease'}: 'n_cells' for overlapping and split diseases"

        cpd = (
            self._adata.obs["disease"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        cpd_split: dict[str, int] = {}  
        values = self._adata.obs["disease"].dropna().astype(str)

        for value in values:
            keys = [key.strip() for key in value.split("||") if key.strip()]

            for key in keys:
                if key == "":
                    continue

                cpd_split[key] = cpd_split.get(key, 0) + 1

        return cpd, cpd_split


    def get_donor_disease_cells(self) -> dict[str, dict[str, str | int]]:
        "returns dict of {'donor': {'disease': disease name}, {n_cells: cell amount}}"

        obs = self._adata.obs[["donor_id", "disease"]].copy()
        obs["disease"] = obs["disease"].astype(str)

        grouped = obs.groupby(["donor_id", "disease"], observed=True).size()

        donor_data: dict[str, dict[str, str | int]] = {}
        for (donor, disease), count in grouped.items():
            if donor not in donor_data:
                donor_data[donor] = {}

            donor_data[donor] = {
                "disease": disease,
                "n_cells": int(count)
            }

        return donor_data


    def get_donors_per_disease(self) -> tuple[dict[str, int], dict[str, int]]:
        "returns dicts of {'disease': amount of donors} for overlapping and split diseases"
        
        dpd: dict[str, int] = {}
        dpd_split: dict[str, int] = {}
        ddc = self.get_donor_disease_cells()

        for donor_key in ddc.keys():
            disease_key = str(ddc[donor_key]["disease"])
            disease_split = str(ddc[donor_key]["disease"])
            disease_keys_split = [key.strip() for key in disease_split.split("||") if key.strip()]

            dpd[disease_key] = dpd.get(disease_key, 0) + 1

            for key in disease_keys_split:
                dpd_split[key] = dpd_split.get(key, 0) + 1

        return dpd, dpd_split


    def get_citation(self) -> str:
        "returns citation of dataset"
        cit = self._adata.uns["citation"]
        return cit


    def get_all(self):
        "returns the values of all analysis functions"
        nc = self.n_cells
        ng = self.n_genes
        gt = self.gene_types
        ids = self.donor_ids
        cpd, cpds = self.get_cells_per_disease()
        ddc = self.get_donor_disease_cells()
        dpd, dpds = self.get_donors_per_disease()
        
        try:
            cit = self.get_citation()
        except:
            cit = ""

        return nc, ng, gt, ids, cpd, cpds, ddc, dpd, dpds, cit


    # ===========================
    # FUNCTIONS FOR WRITING OUTPUT

    def clear_file(self, path):
        "clears the given file"
        with open(path, "w") as f:
            f.write("")


    def write_n(self, path):
        "writes number of cells, number of genes"
        with open(path, "a") as f:
            f.write("Data amounts:\n\n")
            f.write(f"   Cells:  {self.n_cells}\n")
            f.write(f"   Genes:  {self.n_genes}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_diseases(self, path):
        "writes unique diseases in dataset"
        with open(path, "a") as f:
            f.write("\nDiseases: \n\n")
            for d in self.diseases:
                f.write(f"{d}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_cells_per_disease(self, path):
        "writes cells per disease with ovelaps"
        with open(path, "a") as f:
            f.write(f"\nDiseases (amount of cells): \n\n")
            cpd, _ = self.get_cells_per_disease()

            for key in cpd.keys():
                f.write(f"  {key}:  {cpd[key]}\n")
            f.write(f"\nTotal: {len(cpd.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_cells_per_disease_split(self, path):
        "writes cells per disease split"
        with open(path, "a") as f:
            f.write(f"\nDiseases (amount of cells) - split: \n\n")
            _, cpds = self.get_cells_per_disease()

            for key in cpds.keys():
                f.write(f"  {key}:  {cpds[key]}\n")
            f.write(f"\nTotal: {len(cpds.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_donors_per_disease(self, path):
        "writes donors per disease with ovelaps"
        with open(path, "a") as f:
            f.write(f"\nDonor amounts per disease: \n\n")
            dpd, _ = self.get_donors_per_disease()

            for key in dpd.keys():
                f.write(f"  {key}:  {dpd[key]} donors\n")
            f.write(f"\nTotal: {len(dpd.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_donors_per_disease_split(self, path):
        "writes donors per disease split"
        with open(path, "a") as f:
            f.write(f"\nDonor amounts per disease - split: \n\n")
            _, dpds = self.get_donors_per_disease()

            for key in dpds.keys():
                f.write(f"  {key}:  {dpds[key]} donors\n")
            f.write(f"\nTotal: {len(dpds.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_gene_types(self, path):
        "writes gene types and amounts"
        with open(path, "a") as f:
            f.write("\nFeature types: \n\n")
            for key, n in self.gene_types.items():
                f.write(f"   {key}:  {n}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_donor_amount(self, path):
        "writes amount of donors"
        with open(path, "a") as f:
            f.write(f"\nTotal donors: {self.donor_amount}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_citation(self, path):
        "writes dataset citation"
        with open(path, "a") as f:
            f.write(f"\nCitation: {self.citation}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    def write_all(self, path):
        "writes all output"
        self.clear_file(path)

        self.write_n(path)
        self.write_diseases(path)
        self.write_cells_per_disease(path)
        self.write_cells_per_disease_split(path)
        self.write_donors_per_disease(path)
        self.write_donors_per_disease_split(path)
        self.write_gene_types(path)
        self.write_donor_amount(path)

        try:
            self.write_citation(path)
        except:
            pass