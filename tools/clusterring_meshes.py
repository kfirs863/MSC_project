import os
import numpy as np
import open3d as o3d
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt


def load_meshes(obj_file_paths):
    """Load OBJ files as Open3D TriangleMesh objects."""
    meshes = []
    for file_path in obj_file_paths:
        mesh = o3d.io.read_triangle_mesh(file_path)
        meshes.append(mesh)
    return meshes


def extract_geometric_features(meshes):
    """Extract geometric features like volume, surface area, and PCA components."""
    features = []

    for mesh in meshes:
        mesh.compute_vertex_normals()

        # Surface area
        surface_area = mesh.get_surface_area()

        # Volume (can be approximated via convex hull)
        convex_hull, _ = mesh.compute_convex_hull()
        volume = convex_hull.get_volume()

        # Flatten PCA components to a feature vector
        verts = np.asarray(mesh.vertices)
        pca = PCA(n_components=3)
        verts_pca = pca.fit_transform(verts)
        pca_components = verts_pca.flatten()

        # Combine all features
        feature_vector = np.concatenate([[volume, surface_area], pca_components])
        features.append(feature_vector)

    return np.array(features)


def extract_image_features(image_paths):
    """Extract features from contour or depth images."""
    image_features = []

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract HOG features as an example
        hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

        image_features.append(hog_features)

    return np.array(image_features)


def visualize_clustered_meshes(meshes, cluster_labels):
    """Visualize the clustered meshes in 3D, colored by their cluster labels."""
    # Create a unique color for each cluster
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.get_cmap('viridis', num_clusters)

    for i, mesh in enumerate(meshes):
        color = colors(cluster_labels[i])[:3]
        mesh.paint_uniform_color(color)

    # Visualize all the meshes
    o3d.visualization.draw_geometries(meshes, window_name="Clustered Meshes")


def main():
    # Directory containing your OBJ files
    obj_dir = "path_to_your_obj_files"

    # Directory containing the corresponding contour/depth images
    image_dir = "path_to_your_images"

    # List of file paths to the OBJ files and corresponding images
    obj_file_paths = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.obj')]
    image_file_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    # Load the OBJ files as meshes using Open3D
    meshes = load_meshes(obj_file_paths)

    # Extract geometric features
    geometric_features = extract_geometric_features(meshes)

    # Extract image-based features
    image_features = extract_image_features(image_file_paths)

    # Combine the features
    combined_features = np.hstack((geometric_features, image_features))

    # Standardize the features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # Perform clustering
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(combined_features)

    # Visualize or further analyze the cluster labels
    print("Cluster labels:", cluster_labels)

    # Visualize clustered meshes
    visualize_clustered_meshes(meshes, cluster_labels)


if __name__ == "__main__":
    main()
