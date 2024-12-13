# clustering.py

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

def perform_clustering(algorithm, parameters, features):
    """Perform clustering based on the selected algorithm and parameters."""
    if algorithm == 'kmeans':
        n_clusters = parameters.get('n_clusters', 5)
        max_iter = parameters.get('max_iter', 300)
        random_state = parameters.get('random_state', 42)
        clusterer = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    elif algorithm == 'dbscan':
        eps = parameters.get('eps', 0.5)
        min_samples = parameters.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'agglomerative':
        linkage = parameters.get('linkage', 'ward')
        n_clusters = parameters.get('n_clusters', 5)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    elif algorithm == 'spectral':
        n_clusters = parameters.get('n_clusters', 5)
        n_components = parameters.get('n_components', 100)
        clusterer = SpectralClustering(n_clusters=n_clusters, n_components=n_components, random_state=42)
    elif algorithm == 'gmm':
        n_components = parameters.get('n_components', 5)
        covariance_type = parameters.get('covariance_type', 'full')
        clusterer = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    if algorithm != 'gmm':
        cluster_labels = clusterer.fit_predict(features)
    else:
        clusterer.fit(features)
        cluster_labels = clusterer.predict(features)

    return cluster_labels, clusterer
