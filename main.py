from dataset import Dataset

path = "synthetic_data_test.h5ad"

dataset = Dataset(path)
print(dataset.analysis.get_diseases())
print(dataset.analysis.get_n())

dataset.keep_diseases(["normal", "dementia", "schizophrenia"])
print(dataset.analysis.get_diseases())
print(dataset.analysis.get_n())

dataset.analysis.write_all("test_1.txt")

dataset.variable_percentile_subset(0.2)
print(dataset.analysis.get_diseases())
print(dataset.analysis.get_n())

dataset.analysis.write_all("test_2.txt")

dataset.restore_original()
print(dataset.analysis.get_diseases())
print(dataset.analysis.get_n())

dataset.analysis.write_all("test_3.txt")