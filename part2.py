from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
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
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    data, _ = dataset
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_standardized)

    # Inertia attribute gives the SSE (Sum of Squared Errors)
    sse = kmeans.inertia_
    
    return kmeans.labels_, sse



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    data, labels = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)


    dct = answers["2A: blob"] = [data, labels, np.zeros(data.shape[0])]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    # Assuming 'data' is your dataset
    sse_values = []
    for k in range(1, 9):  # From k=1 to k=8
          _, sse = fit_kmeans((data, None), k)  # Using the modified fit_kmeans function
          sse_values.append([k, sse])
    import matplotlib.pyplot as plt

    ks, sses = zip(*sse_values)  # Unpacking the k and SSE values
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sses, '-o')
    plt.title('SSE vs. k')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.xticks(range(1, 9))  # Ensure x-ticks for every k value
    plt.grid()
    plt.show()

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = sse_values


    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # Assuming 'data' is your dataset
    inertia_values = []
    for k in range(1, 9):  # From k=1 to k=8
           _, inertia = fit_kmeans((data, None), k)  # fit_kmeans already returns inertia as the second value
           inertia_values.append([k, inertia])

# Plotting inertia to determine the optimal k using the elbow method
    ks, inertias = zip(*inertia_values)  # Unpacking the k and inertia values
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, '-o')
    plt.title('Inertia vs. k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 9))
    plt.grid()
    plt.show()

# Storing the k, inertia pairs
    dct = answers["2D: inertia plot"] = inertia_values


    optimal_k_previous = 3  
    optimal_k_current = 3  

# Checking if the optimal k's agree
    dct = answers["2D: do ks agree?"] = "yes" 

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
