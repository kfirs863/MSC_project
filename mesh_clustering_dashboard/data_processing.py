# data_processing.py

import os
import numpy as np
import trimesh
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import yaml
from torchvision.models import ResNet50_Weights


def initialize_resnet_model():
    """Initialize a pre-trained ResNet model for feature extraction."""
    # Load the pre-trained ResNet model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the last fully connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    # Set the model to evaluation mode
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
                contour_images = [cv2.imread(contour_file, cv2.IMREAD_GRAYSCALE) for contour_file in
                                  contour_image_files]
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
    degree_matrix = csr_matrix((adjacency_matrix.sum(axis=1).A1, (vertex_indices, vertex_indices)),
                               shape=(vertex_count, vertex_count))

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
    """Extract geometric features from meshes in a predefined order."""
    features = []
    feature_data = []
    feature_names = [
        'surface_area',
        'sa_to_bbox_volume_ratio',
        'num_vertices',
        'num_faces',
        'mean_curvature',
        'std_curvature',
        'mean_edge_length',
        'std_edge_length',
        'mean_face_area',
        'std_face_area'
    ]

    for mesh in meshes:
        surface_area = mesh.area
        bbox_size = mesh.bounds[1] - mesh.bounds[0]
        sa_to_bbox_volume_ratio = surface_area / (np.prod(bbox_size) if np.prod(bbox_size) != 0 else 1)

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)

        curvature = compute_laplacian_curvature_vectorized(mesh)
        mean_curvature = np.mean(curvature)
        std_curvature = np.std(curvature)

        edges = mesh.edges_unique_length
        mean_edge_length = np.mean(edges)
        std_edge_length = np.std(edges)

        face_areas = mesh.area_faces
        mean_face_area = np.mean(face_areas) if len(face_areas) > 0 else 0
        std_face_area = np.std(face_areas) if len(face_areas) > 0 else 0

        # Ensure all features are 1D arrays for concatenation
        feature_vector = np.concatenate([
            [surface_area],
            [sa_to_bbox_volume_ratio],
            [num_vertices, num_faces],
            [mean_curvature, std_curvature],
            [mean_edge_length, std_edge_length],
            [mean_face_area, std_face_area]
        ])

        features.append(feature_vector)
        feature_data.append({
            'mesh': mesh,
            'surface_area': surface_area,
            'sa_to_bbox_volume_ratio': sa_to_bbox_volume_ratio,
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'curvature': curvature,            # Added line
            'mean_curvature': mean_curvature,
            'std_curvature': std_curvature,
            'edge_lengths': edges,             # Added line
            'mean_edge_length': mean_edge_length,
            'std_edge_length': std_edge_length,
            'face_areas': face_areas,          # Added line
            'mean_face_area': mean_face_area,
            'std_face_area': std_face_area
        })

    return np.vstack(features), feature_data, feature_names

def extract_image_features(images, model, transform):
    """Extract features from images (ndarray) using a pre-trained ResNet model."""
    image_features = []

    for image in images:
        if isinstance(image, np.ndarray):
            # Convert grayscale to RGB if necessary
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and apply transformations
            pil_image = Image.fromarray(image)
            input_tensor = transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                features = model(input_tensor).squeeze().numpy()

            image_features.append(features)
        else:
            print(f"Warning: Image is not a valid ndarray. Skipping.")

    return np.array(image_features)



def extract_combined_contour_features(contour_images_list, model, transform):
    """Extract and combine features from multiple contour images."""
    combined_contour_features = []

    for contour_images in contour_images_list:
        # Extract features from each contour image using the pre-trained ResNet
        contour_features = extract_image_features(contour_images, model, transform)
        # Combine the features from multiple contour images (e.g., by averaging or concatenation)
        combined_features = np.mean(contour_features, axis=0)  # Here, we're using averaging
        combined_contour_features.append(combined_features)

    return np.array(combined_contour_features)
