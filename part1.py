import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from myplots import plot_part1C
import pickle

# Fit k-means function
def fit_kmeans(dataset, n_clusters, random_state=42):
    data, _ = dataset
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=random_state)
    kmeans.fit(data_standardized)
    return kmeans.labels_

# Main computation function
def compute():
    answers = {}
    n_samples = 100
    random_state = 42

# Loading the datasets as per the specified requirements
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    blobs_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso_data, aniso_labels = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    aniso = (np.dot(aniso_data, [[0.6, -0.6], [-0.4, 0.8]]), aniso_labels)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

# Storing the datasets in the dictionary
    datasets_dict = {
    'nc': noisy_circles,
    'nm': noisy_moons,
    'bvv': blobs_varied,
    'add': aniso,
    'b': blobs
}


    answers = {
    "1A: datasets": datasets_dict
}
    
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    k_values = [2, 3, 5, 10]
    random_states = [42, 43, 44, 45]
    kmeans_results = {}

    # Applying k-means and storing results
    for name, dataset in datasets_dict.items():
        for k in k_values:
            for state in random_states:
                kmeans_results.setdefault(name, {}).setdefault(k, {})[state] = fit_kmeans(dataset, k, state)

    # Placeholder for analysis
    dct = answers["1C: cluster successes"] = {
        'bvv': [3, 5],
        'b': [2, 3, 5]
    } 
    answers["1C: cluster failures"] = {"add": [2, 3, 5, 10]}
    dct = answers["1C: cluster failures"] = ['nc', 'nm', 'add']
    # Extracting results for a selected random state for demonstration
    selected_random_state = 44
    kmeans_dct = {}
    for name, (data, _) in datasets_dict.items():
        clustering_results = {}
        for k in k_values:
            if selected_random_state in kmeans_results[name][k]:
                clustering_results[k] = kmeans_results[name][k][selected_random_state]
        kmeans_dct[name] = ([data, None], clustering_results)

    # Plotting results for the selected random state
    plot_part1C(kmeans_dct, "clustering_results.pdf")

    answers["1D: datasets sensitive to initialization"] = ["nc", "nm"]

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
