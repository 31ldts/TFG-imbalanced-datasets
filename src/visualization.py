import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan

from config.parameters import *

def elbow_method(X, max_k=10):
    """
    Elbow method for KMeans using inertia (SSE).
    """
    inertias = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.title(ELBOW_METHOD_TITLE)
    plt.xlabel(ELBOW_X_LABEL)
    plt.ylabel(ELBOW_Y_LABEL)
    plt.grid(True)

    output_path = os.path.join(PATH_IMAGES, f"{ELBOW_METHOD_FILENAME}.png")
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.show()


def silhouette_method(
    X,
    method="kmeans",
    max_k=10,
    eps_values=None,
    min_cluster_sizes=None,
    random_state=DEFAULT_RANDOM_STATE
):
    """
    Searches for the optimal clustering configuration using Silhouette Score.

    - kmeans, gmm, bayesian_gmm, agglomerative, spectral → iterate over n_clusters
    - dbscan → iterate over eps values
    - hdbscan → iterate over min_cluster_size
    """

    scores = []
    configurations = []

    # --------------------------------------------------------
    # K-based methods
    # --------------------------------------------------------
    if method in SILHOUETTE_METHODS_K_BASED:
        for k in range(2, max_k + 1):

            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=random_state)
                labels = model.fit_predict(X)

            elif method == "gmm":
                model = GaussianMixture(n_components=k, random_state=random_state)
                labels = model.fit_predict(X)

            elif method == "bayesian_gmm":
                model = BayesianGaussianMixture(n_components=k, random_state=random_state)
                labels = model.fit_predict(X)

            elif method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)

            elif method == "spectral":
                model = SpectralClustering(
                    n_clusters=k,
                    random_state=random_state,
                    assign_labels="discretize"
                )
                labels = model.fit_predict(X)

            score = (
                silhouette_score(X, labels)
                if len(set(labels)) > 1
                else np.nan
            )

            scores.append(score)
            configurations.append(k)

        x_label = X_LABEL_NUM_CLUSTERS

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------
    elif method == "dbscan":
        if eps_values is None:
            eps_values = DEFAULT_EPS_VALUES

        for eps in eps_values:
            model = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
            labels = model.fit_predict(X)

            mask = labels != -1
            score = (
                silhouette_score(X[mask], labels[mask])
                if mask.sum() > 1 and len(set(labels[mask])) > 1
                else np.nan
            )

            scores.append(score)
            configurations.append(eps)

        x_label = X_LABEL_EPS

    # --------------------------------------------------------
    # HDBSCAN
    # --------------------------------------------------------
    elif method == "hdbscan":
        if min_cluster_sizes is None:
            min_cluster_sizes = DEFAULT_MIN_CLUSTER_SIZES

        for size in min_cluster_sizes:
            model = hdbscan.HDBSCAN(min_cluster_size=size)
            labels = model.fit_predict(X)

            mask = labels != -1
            score = (
                silhouette_score(X[mask], labels[mask])
                if mask.sum() > 1 and len(set(labels[mask])) > 1
                else np.nan
            )

            scores.append(score)
            configurations.append(size)

        x_label = X_LABEL_MIN_CLUSTER_SIZE

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    title = SILHOUETTE_TITLE_TEMPLATE.format(method=method)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.plot(configurations, scores, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(SILHOUETTE_Y_LABEL)
    plt.grid(True)

    filename = title.replace(" ", "_").replace("(", "").replace(")", "")
    output_path = os.path.join(PATH_IMAGES, f"{filename}.png")

    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.show()

    return configurations, scores
