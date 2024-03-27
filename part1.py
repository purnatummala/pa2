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
        if isinstance(dataset, tuple) and len(dataset) == 2:
             data, labels = dataset  # Assuming dataset is a tuple (data, labels)
        else:
             data = dataset  # Assuming dataset is just the data
             labels = None  # Unpacking the dataset into data and labels
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)  # Standardizing the data
    
    # Applying KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=62)
        kmeans.fit(data_scaled)
    
        return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    n_samples = 100
    random_state = 42

    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    blobs_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    
    aniso_data, _ = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso_data = np.dot(aniso_data, transformation)
    
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

   
   
    dct = answers["1A: datasets"] = {
    "noisy_circles": noisy_circles[0],
    "noisy_moons": noisy_moons[0],
    "blobs_varied": blobs_varied[0],
    "anisotropic": aniso_data,
    "blobs": blobs[0],
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
    k_values = [2, 3,5,10]
    dataset_1 = {
         'noisy_circles': noisy_circles[0],  # Assuming noisy_circles = (data, labels)
         'noisy_moons': noisy_moons[0],
         'blobs_varied': blobs_varied[0],
         'anisotropic': aniso_data,
         'blobs': blobs[0],
        }

    l = 5
    
    clustering_results = {}

    for name, data in dataset_1.items():
            clustering_results[name] = {}
            for k in k_values:
                 clustering_results[name][k] = fit_kmeans(data, k)


    kmeans_dct = {name: ([data, None], clustering_results[name]) for name, data in dataset_1.items()}


    plot_part1C(kmeans_dct, "report.pdf")
    
     
    
    
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
   
    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = {"nc","nm","bvv","add","b"}

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
