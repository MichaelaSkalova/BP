class Analyzer():

    def __init__(self, dataset):
        self.adata = dataset.get_adata()
        self.__init_all()

    def get_n(self):
        '''returns n_cells, n_genes'''
        n_obs = self.adata.n_obs
        n_vars = self.adata.n_vars
        return n_obs, n_vars


    def get_diseases(self):
        data = self.adata.obs["disease"]
        unique_diseases = data.dropna().unique().tolist()
        return unique_diseases


    def get_cell_diseases(self):
        return self.adata.obs["disease"].to_numpy()


    def get_cells_per_disease(self):
        '''returns dicts of {"disease": n_cells} 
        for overlapping diseases and split diseases'''

        cpd = (
            self.adata.obs["disease"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        cpd_split = {}  
        values = self.adata.obs["disease"].dropna().astype(str)

        for value in values:
            keys = [key.strip() for key in value.split("||") if key.strip()]

            for key in keys:
                if key == "":
                    continue

                cpd_split[key] = cpd_split.get(key, 0) + 1

        return cpd, cpd_split


    def get_donor_disease_cells(self):
        '''returns dict of {"donor": {"disease": disease name}, {"n_cells: cell amount}}'''

        obs = self.adata.obs[["donor_id", "disease"]].copy()
        obs["disease"] = obs["disease"].astype(str)

        grouped = obs.groupby(["donor_id", "disease"], observed=True).size()

        donor_data = {}
        for (donor, disease), count in grouped.items():
            if donor not in donor_data:
                donor_data[donor] = {}

            donor_data[donor] = {
                "disease": disease,
                "n_cells": int(count)
            }

        return donor_data


    def get_donors_per_disease(self):
        '''returns dicts of {"disease": amount of donors}
        for overlapping and split diseases'''
        dpd = {}
        dpd_split = {}
        ddc = self.donor_disease_cells

        for donor_key in ddc.keys():
            disease_key = str(ddc[donor_key]["disease"])
            disease_split = str(ddc[donor_key]["disease"])
            disease_keys_split = [key.strip() for key in disease_split.split("||") if key.strip()]

            dpd[disease_key] = dpd.get(disease_key, 0) + 1

            for key in disease_keys_split:
                dpd_split[key] = dpd_split.get(key, 0) + 1

        return dpd, dpd_split


    def get_gene_types(self):
        gene_types = self.adata.var["feature_type"].value_counts(dropna=False)
        return gene_types


    def get_donor_ids(self):
        data = self.adata.obs["donor_id"]
        unique_ids = data.dropna().unique().tolist()

        return unique_ids


    # def get_citation(self):
    #     cit = self.adata.uns["citation"]
    #     return cit


    def get_donor_amount(self):
        return self.donor_amount


    def __init_all(self):
        self.n_cells, self.n_genes = self.get_n()
        self.diseases = self.get_diseases()
        self.cell_diseases = self.get_cell_diseases()

        self.gene_types = self.get_gene_types()
        self.donor_ids = self.get_donor_ids()
        self.donor_amount = len(self.donor_ids)

        self.cells_per_disease, self.cells_per_disease_split = self.get_cells_per_disease()
        self.donor_disease_cells = self.get_donor_disease_cells()
        self.donors_per_disease, self.donors_per_disease_split = self.get_donors_per_disease()

        # self.citation = self.get_citation()


    def get_all(self):
        nc = self.n_cells
        ng = self.n_genes
        gt = self.gene_types
        ids = self.donor_ids
        cpd = self.cells_per_disease
        cpds = self.cells_per_disease_split
        ddc = self.donor_disease_cells
        dpd = self.donors_per_disease
        dpds = self.donors_per_disease_split
        # cit = self.citation

        return nc, ng, gt, ids, cpd, cpds, ddc, dpd, dpds


    # ===========================
    # FUNCTIONS FOR WRITING OUTPUT

    def clear_file(self, path):
        with open(path, "w") as f:
            f.write("")

    def write_n(self, path):
        with open(path, "a") as f:
            f.write("Data amounts:\n\n")
            f.write(f"   Cells:  {self.n_cells}\n")
            f.write(f"   Genes:  {self.n_genes}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_diseases(self, path):
        with open(path, "a") as f:
            f.write("\nDiseases: \n\n")
            for d in self.diseases:
                f.write(f"{d}\n")
        f.write(f"\n" + "=" * 40 + "\n")

    def write_cells_per_disease(self, path):
        with open(path, "a") as f:
            f.write(f"\nDiseases (amount of cells): \n\n")
            cpd = self.cells_per_disease

            for key in cpd.keys():
                f.write(f"  {key}:  {cpd[key]}\n")
            f.write(f"\nTotal: {len(cpd.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_cells_per_disease_split(self, path):
        with open(path, "a") as f:
            f.write(f"\nDiseases (amount of cells) - split: \n\n")
            cpds = self.cells_per_disease_split

            for key in cpds.keys():
                f.write(f"  {key}:  {cpds[key]}\n")
            f.write(f"\nTotal: {len(cpds.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_donors_per_disease(self, path):
        with open(path, "a") as f:
            f.write(f"\nDonor amounts per disease: \n\n")
            dpd = self.donors_per_disease

            for key in dpd.keys():
                f.write(f"  {key}:  {dpd[key]} donors\n")
            f.write(f"\nTotal: {len(dpd.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_donors_per_disease_split(self, path):
        with open(path, "a") as f:
            f.write(f"\nDonor amounts per disease - split: \n\n")
            dpds = self.donors_per_disease_split

            for key in dpds.keys():
                f.write(f"  {key}:  {dpds[key]} donors\n")
            f.write(f"\nTotal: {len(dpds.keys())}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_gene_types(self, path):
        with open(path, "a") as f:
            f.write("\nFeature types: \n\n")
            for key, n in self.gene_types.items():
                f.write(f"   {key}:  {n}\n")
            f.write(f"\n" + "=" * 40 + "\n")

    def write_donor_amount(self, path):
        with open(path, "a") as f:
            f.write(f"\nTotal donors: {self.donor_amount}\n")
            f.write(f"\n" + "=" * 40 + "\n")


    # def write_citation(self, path):
    #     with open(path, "a") as f:
    #         f.write(f"\nCitation: {self.citation}\n")
    #         f.write(f"\n" + "=" * 40 + "\n")


    def write_all(self, path):
        self.clear_file(path)

        self.write_n(path)
        self.write_diseases(path)
        self.write_cells_per_disease(path)
        self.write_cells_per_disease_split(path)
        self.write_donors_per_disease(path)
        self.write_donors_per_disease_split(path)
        self.write_gene_types(path)
        self.write_donor_amount(path)
        # self.write_citation(path)