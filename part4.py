import time
import warnings
from myplots import plot_part1C
import myplots as myplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
import scipy.io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn import datasets
# import plotly.figure_factory as ff
import math
from myplots import plot_part1C
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, n_clusters):
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    
    # Initialize AgglomerativeClustering estimator
    model = AgglomerativeClustering(n_clusters=n_clusters)
    
    # Fit the model and return labels
    labels = model.fit_predict(dataset_scaled)
    return labels

def fit_modified(data, linkage_type):
    # Calculate linkage matrix
    Z = linkage(data[0], linkage_type)

    # Calculate distances between merges
    distances = []
    for i in range(len(Z) - 1):
        merge_dist = Z[i + 1, 2] - Z[i, 2]
        distances.append(merge_dist)
    
    # Determine distance threshold
    distance_threshold = np.max(distances)

    # Initialize AgglomerativeClustering model
    model = AgglomerativeClustering(n_clusters=None, linkage=linkage_type, distance_threshold=distance_threshold)
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[0])

    # Fit the model
    model.fit(standardized_data, data[1])
    labels = model.labels_
    
    return labels



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
    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    data = scipy.io.loadmat('hierarchical_toy_data.mat')['X']

# Perform hierarchical clustering
    n_clusters = 3  # Set the number of clusters
    labels = fit_hierarchical_cluster(data, n_clusters)

# Create a linkage matrix using single linkage
    Z = linkage(data, method='single')

# Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Dendrogram with Single Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    datasets_dict = {
    'nc': noisy_circles,
    'nm': noisy_moons,
    'bvv': blobs_varied,
    'add': aniso,
    'b': blobs
}
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = datasets_dict

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    import myplots

    hierarchical_results = {}

    for dataset_key, dataset_data_labels in answers['4A: datasets'].items():
        dataset_results = []
        linkage_results = {}
    
        for linkage_type in ['single', 'complete', 'ward', 'average']:
           cluster_preds = fit_hierarchical_cluster(dataset_data_labels[0], 2)  # Pass dataset and number of clusters
           linkage_results[linkage_type] = cluster_preds
    
        dataset_results.append((dataset_data_labels[0], dataset_data_labels[1]))
        dataset_results.append(linkage_results)
        hierarchical_results[dataset_key] = dataset_results



# Call plot_part1C function from myplots.py
    plot_part1C(hierarchical_results, 'Part4_B.pdf')


    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    modified_function_results = {}

    for dataset_key, dataset_data_labels in answers['4A: datasets'].items():
         dataset_results = []
         linkage_results = {}
    
         for linkage_type in ['single', 'complete', 'ward', 'average']:
            cluster_preds = fit_modified((dataset_data_labels[0], dataset_data_labels[1]), linkage_type)  # Correct function call
            linkage_results[linkage_type] = cluster_preds
    
         dataset_results.append((dataset_data_labels[0], dataset_data_labels[1]))
         dataset_results.append(linkage_results)
         modified_function_results[dataset_key] = dataset_results

# Call plot_part1C function from myplots.py
    plot_part1C(modified_function_results, 'Part4_C.pdf')




    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
