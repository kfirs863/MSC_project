# main_script.py

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import trimesh
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from PIL import Image

# Import visualization functions
from visualizations import *

# Load a pre-trained ResNet model for image feature extraction
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Remove the final classification layer to get feature vectors
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define the image transformation (resize, normalize to match ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),          # Convert the image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def normalize_mesh(mesh, only_scale=False):
    """Normalize the mesh for translation, rotation, and scale invariance."""
    # Create a copy of the mesh to avoid modifying the original
    mesh_copy = mesh.copy()

    if not only_scale:
        # Move the centroid to the origin (translation invariance)
        centroid = mesh_copy.vertices.mean(axis=0)
        mesh_copy.vertices -= centroid

        # Apply PCA to align principal axes (rotation invariance)
        pca = PCA(n_components=3)
        pca.fit(mesh_copy.vertices)
        mesh_copy.vertices = np.dot(mesh_copy.vertices, pca.components_.T)

    # Scale the mesh to fit within a unit sphere (scale invariance)
    scale = np.linalg.norm(mesh_copy.vertices, axis=1).max()
    mesh_copy.vertices /= scale

    return mesh_copy

def load_meshes_and_images_from_folders(datasets_folder):
    """Load meshes and corresponding images from a folder hierarchy, and track subfolder names."""
    meshes = []
    normalized_meshes = []
    depth_images = []
    contour_images_list = []
    skeleton_images = []
    subfolder_names = []

    # Traverse each subfolder in the datasets folder
    for subfolder_name in os.listdir(datasets_folder):
        subfolder_path = os.path.join(datasets_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            ply_file = None
            depth_image_file = None
            contour_image_files = []
            skeleton_image_file = None

            # Traverse files within each subfolder to find mask.ply, depth_profile.png, contour_*.png, and skeleton.png
            for file_name in os.listdir(subfolder_path):
                if file_name == 'mask.ply':  # Mesh file name is 'mask.ply'
                    ply_file = os.path.join(subfolder_path, file_name)
                elif file_name == 'depth_profile.png':  # Depth profile image
                    depth_image_file = os.path.join(subfolder_path, file_name)
                elif file_name.startswith('contour_') and file_name.endswith('.png'):  # Contour images
                    contour_image_files.append(os.path.join(subfolder_path, file_name))
                elif file_name == 'skeleton.png':  # Skeleton image
                    skeleton_image_file = os.path.join(subfolder_path, file_name)

            # Ensure all necessary files are found
            if ply_file and depth_image_file and contour_image_files and skeleton_image_file:
                # Load the mesh using Trimesh
                mesh = trimesh.load(ply_file)
                scaled_mesh = normalize_mesh(mesh, only_scale=True)
                normalized_mesh = normalize_mesh(mesh)
                meshes.append(scaled_mesh)
                normalized_meshes.append(normalized_mesh)

                # Load the depth profile image
                depth_image = cv2.imread(depth_image_file, cv2.IMREAD_GRAYSCALE)
                depth_images.append(depth_image)

                # Load all contour images
                contour_images = [cv2.imread(contour_file, cv2.IMREAD_GRAYSCALE) for contour_file in contour_image_files]
                contour_images_list.append(contour_images)

                # Load the skeleton image
                skeleton_image = cv2.imread(skeleton_image_file, cv2.IMREAD_GRAYSCALE)
                skeleton_images.append(skeleton_image)

                # Track the subfolder name for tracing back later
                subfolder_names.append(subfolder_name)
            else:
                print(f"Skipping {subfolder_name} due to missing files.")
                continue

    return meshes, normalized_meshes, depth_images, contour_images_list, skeleton_images, subfolder_names


def compute_laplacian_curvature_vectorized(mesh):
    """Compute mean curvature using a vectorized Laplace-Beltrami approach with sparse matrix."""
    vertex_count = len(mesh.vertices)
    vertex_indices = np.arange(vertex_count)

    # Build sparse adjacency matrix
    data = []
    rows = []
    cols = []

    for vi in vertex_indices:
        neighbors = mesh.vertex_neighbors[vi]
        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            data.extend([1] * num_neighbors)
            rows.extend([vi] * num_neighbors)
            cols.extend(neighbors)

    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))

    # Degree matrix: diagonal matrix with the degree of each vertex
    degree_matrix = csr_matrix((adjacency_matrix.sum(axis=1).A1, (vertex_indices, vertex_indices)), shape=(vertex_count, vertex_count))

    # Laplacian Matrix: L = D - A
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Compute the Laplacian vector for each vertex
    laplacian = laplacian_matrix.dot(mesh.vertices)

    # Normalize by the degree to get the mean curvature vector at each vertex
    degrees = adjacency_matrix.sum(axis=1).A1  # Get the degree of each vertex
    laplacian /= degrees[:, np.newaxis]  # Avoid division by zero by handling isolated vertices

    # Compute the norm of the Laplacian vector, which corresponds to mean curvature magnitude
    curvature = np.linalg.norm(laplacian, axis=1)

    # Handle isolated vertices (if any)
    curvature[np.isnan(curvature)] = 0

    return curvature


def extract_geometric_features(meshes):
    """Extract enhanced geometric features from meshes without volume-dependent features."""
    features = []
    feature_data = []

    for mesh in meshes:
        surface_area = mesh.area
        bounding_box = mesh.bounds
        bbox_size = bounding_box[1] - bounding_box[0]
        bbox_aspect_ratios = [
            bbox_size[0] / bbox_size[1] if bbox_size[1] != 0 else 0,
            bbox_size[1] / bbox_size[2] if bbox_size[2] != 0 else 0,
            bbox_size[0] / bbox_size[2] if bbox_size[2] != 0 else 0
        ]
        bbox_volume = np.prod(bbox_size) if np.prod(bbox_size) != 0 else 1
        sa_to_bbox_volume_ratio = surface_area / bbox_volume

        verts = mesh.vertices
        pca = PCA(n_components=3)
        pca.fit(verts)
        eigenvalues = pca.explained_variance_
        eigenvalues_ratio = eigenvalues / np.sum(eigenvalues)
        first_to_second_axis_ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] != 0 else float('inf')

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)

        curvature = compute_laplacian_curvature_vectorized(mesh)
        mean_curvature = np.mean(curvature)
        std_curvature = np.std(curvature)

        edges = mesh.edges_unique_length
        mean_edge_length = np.mean(edges)
        std_edge_length = np.std(edges)

        face_areas = mesh.area_faces
        mean_face_area = np.mean(face_areas)
        std_face_area = np.std(face_areas)

        feature_vector = np.concatenate([
            [surface_area],
            bbox_size,
            bbox_aspect_ratios,
            [sa_to_bbox_volume_ratio],
            eigenvalues,
            eigenvalues_ratio,
            [first_to_second_axis_ratio],
            [num_vertices, num_faces],
            [mean_curvature, std_curvature],
            [mean_edge_length, std_edge_length],
            [mean_face_area, std_face_area]
        ])

        features.append(feature_vector)
        feature_data.append({
            'mesh': mesh,
            'surface_area': surface_area,
            'bbox_size': bbox_size,
            'aspect_ratios': bbox_aspect_ratios,
            'sa_to_bbox_volume_ratio': sa_to_bbox_volume_ratio,
            'pca_components': pca.components_,
            'pca_mean': pca.mean_,
            'eigenvalues': eigenvalues,
            'eigenvalues_ratio': eigenvalues_ratio,
            'first_to_second_axis_ratio': first_to_second_axis_ratio,
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'curvature': curvature,
            'edge_lengths': edges,
            'face_areas': face_areas
        })

    return np.vstack(features), feature_data

def extract_image_features(images):
    """Extract features from images (ndarray) using a pre-trained ResNet model."""
    image_features = []

    for image in images:
        # Ensure the image is a valid ndarray
        if isinstance(image, np.ndarray):
            # OpenCV loads images as BGR, but ResNet expects RGB, so we need to convert the color channels
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image (ndarray) to a PIL Image
            pil_image = Image.fromarray(image)

            # Apply transformations and extract features using the pre-trained ResNet model
            input_tensor = transform(pil_image).unsqueeze(0)  # Add a batch dimension
            with torch.no_grad():  # Disable gradient computation
                features = model(input_tensor).squeeze().numpy()  # Extract ResNet features

            image_features.append(features)
        else:
            print(f"Warning: Image is not a valid ndarray. Skipping.")

    return np.array(image_features)

def extract_combined_contour_features(contour_images_list):
    """Extract and combine features from multiple contour images."""
    combined_contour_features = []

    for contour_images in contour_images_list:
        # Extract features from each contour image using the pre-trained ResNet
        contour_features = extract_image_features(contour_images)
        # Combine the features from multiple contour images (e.g., by averaging or concatenation)
        combined_features = np.mean(contour_features, axis=0)  # Here, we're using averaging
        combined_contour_features.append(combined_features)

    return np.array(combined_contour_features)


def main():
    # Directory containing subfolders with datasets
    datasets_folder = "/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets"  # Update this path

    # Load the meshes and corresponding images from subfolders
    meshes, normalized_meshes, depth_images, contour_images_list, skeleton_images, subfolder_names = load_meshes_and_images_from_folders(datasets_folder)

    # Extract geometric features from meshes
    geometric_features, feature_data = extract_geometric_features(normalized_meshes)

    # Extract and combine features from contour images
    contour_features = extract_combined_contour_features(contour_images_list)

    # Extract features from depth images
    depth_features = extract_image_features(depth_images)

    # Extract features from skeleton images
    skeleton_features = extract_image_features(skeleton_images)

    # Combine all features
    combined_features = np.hstack((geometric_features, contour_features))

    # Standardize the combined features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # Apply PCA to reduce the dimensionality of the combined features
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    reduced_features = pca.fit_transform(combined_features)
    print(f"Reduced features shape: {reduced_features.shape}")

    # Perform clustering on the reduced features
    num_clusters = 5  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, n_init=50, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Calculate and print clustering evaluation metrics
    silhouette_avg = silhouette_score(reduced_features, cluster_labels)
    db_score = davies_bouldin_score(reduced_features, cluster_labels)
    inertia = kmeans.inertia_

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"KMeans Inertia: {inertia}")

    # --- t-SNE Visualization ---
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('t-SNE Clustering Results')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Visualize clustered meshes with their subfolder names
    visualize_clustered_meshes(meshes, cluster_labels)

    # Extract data for visualization
    surface_areas = [data['surface_area'] for data in feature_data]
    aspect_ratios = [data['aspect_ratios'] for data in feature_data]
    sa_to_volume_ratios = [data['sa_to_bbox_volume_ratio'] for data in feature_data]
    num_vertices = [data['num_vertices'] for data in feature_data]
    num_faces = [data['num_faces'] for data in feature_data]

    # Visualize surface area
    # visualize_surface_area(meshes, surface_areas)

    # Visualize aspect ratios
    visualize_aspect_ratios(aspect_ratios, subfolder_names)

    # Visualize the first to second principal axis ratio
    visualize_first_to_second_axis_ratios(feature_data, subfolder_names)

    # Visualize surface area to volume ratio
    visualize_sa_to_volume_ratio(sa_to_volume_ratios, subfolder_names)

    # Visualize vertices and faces
    visualize_vertices_faces(num_vertices, num_faces, subfolder_names)

    # For each mesh, visualize curvature and face areas
    for data, name in zip(feature_data, subfolder_names):
        visualize_curvature(data['mesh'], data['curvature'])
        visualize_face_areas(data['mesh'])
        visualize_edge_lengths(data['edge_lengths'], name)

if __name__ == "__main__":
    main()
