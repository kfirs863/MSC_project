import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

def perform_clustering(algorithm, params, features):
    """
    Perform clustering using the specified algorithm and parameters.
    Returns (labels, clusterer).
    """
    if algorithm == 'kmeans':
        n_clusters = params.get('n_clusters', 5)
        max_iter = params.get('max_iter', 20)
        random_state = params.get('random_state', None)
        clusterer = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        cluster_labels = clusterer.fit_predict(features)
    elif algorithm == 'dbscan':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(features)
    elif algorithm == 'agglomerative':
        n_clusters = params.get('n_clusters', 5)
        linkage = params.get('linkage', 'ward')
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = clusterer.fit_predict(features)
    elif algorithm == 'spectral':
        n_clusters = params.get('n_clusters', 5)
        n_components = params.get('n_components', 100)
        clusterer = SpectralClustering(n_clusters=n_clusters, n_components=n_components, affinity='nearest_neighbors')
        cluster_labels = clusterer.fit_predict(features)
    elif algorithm == 'gmm':
        n_components = params.get('n_components', 5)
        covariance_type = params.get('covariance_type', 'full')
        clusterer = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        cluster_labels = clusterer.fit_predict(features)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return cluster_labels, clusterer
