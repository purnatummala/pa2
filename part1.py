import myplots as myplt
import time
import warnings
import numpy as np
from sklearn import datasets
from myplots import plot_part1C
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
         data, _ = dataset

    # Standardizing the data
         scaler = StandardScaler()
         data_standardized = scaler.fit_transform(data)

    # Initializing and fitting the KMeans model
         kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
         kmeans.fit(data_standardized)

    # Returning the labels
         return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    random_state = 42
    n_samples = 100

# Generating the datasets as per the provided specifications
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

# Anisotropicly transforming the data
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso[0], transformation), aniso[1])
   
    datasets_dict = dct = answers["1A: datasets"] = {
    'nc': noisy_circles,
    'nm': noisy_moons,
    'bvv': varied,
    'abb': aniso,
    'b': blobs
    }

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    k_values = [2, 3, 5, 10]
    kmeans_results = {}

# Iterate over each dataset and each k value, apply KMeans and store the results
    for dataset_name, dataset in datasets_dict.items():
        data, _ = dataset
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
    
        kmeans_results[dataset_name] = (dataset, {})
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='random', random_state=42)
            kmeans.fit(data_standardized)
            kmeans_results[dataset_name][1][k] = kmeans.labels_



    kmeans_dct = {name: ([data, None], kmeans_results[name][1]) for name, (data, _) in datasets_dict.items()}

    plot_part1C(kmeans_dct, "clustering_results.pdf")
    
     
    
    
    dct = answers["1C: cluster successes"] =  {
        "bvv": [2, 3,5],  # blobs_varied, k-means succeeds for all k
        "b": [2, 3],    # blobs, k-means succeeds for all k
        "nm": [2,3,5],
        "nc":[2,3,5]              # noisy_moons, k-means may be considered successful for k=2
    }
    
    dct = answers["1C: cluster failures"] ={
         "bvv": [10],
         "b": [5, 10],    
        "nc": [10],  # noisy_circles, k-means fails for all k
        "add": [2, 3, 5, 10], # anisotropic, k-means fails for all k
        "nm": [10]              
    }
   
    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
   

    kmeans_results = {}
    random_states = [42, 43, 44, 45]  # Example random states
    k_values = [2, 3]

    for random_state in random_states:
         for dataset_name, dataset in datasets_dict.items():
             data, _ = dataset
             scaler = StandardScaler()
             data_standardized = scaler.fit_transform(data)
        
             for k in k_values:
                kmeans = KMeans(n_clusters=k, init='random', random_state=random_state)
                kmeans.fit(data_standardized)
                if dataset_name not in kmeans_results:
                     kmeans_results[dataset_name] = {}
                if k not in kmeans_results[dataset_name]:
                     kmeans_results[dataset_name][k] = {}
                kmeans_results[dataset_name][k][random_state] = kmeans.labels_

# Now we construct kmeans_dct with the correct structure
    selected_random_state = 44  # Or any other random state from the list
    kmeans_dct = {}

    for name, (data, _) in datasets_dict.items():
    # Extract the clustering results for the selected random state and all k values
       clustering_results = {k: kmeans_results[name][k][selected_random_state] for k in k_values if selected_random_state in kmeans_results[name][k]}
       kmeans_dct[name] = ([data, None], clustering_results)

# Assuming plot_part1C is defined and works as expected
    plot_part1C(kmeans_dct, "report.pdf")

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = {}

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
