import os
import numpy as np
import trimesh
from PIL import Image
import yaml
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

from tools.utils import find_rotation_matrix  # or your own utility if needed

def load_feature_descriptions(yaml_path='feature_descriptions.yaml'):
    """
    Load a YAML or dictionary describing each feature.
    If you don't have an actual file, you can return a dict manually.
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        # fallback: return an empty or default dict
        return {
            "surface_area": "Surface area of the mesh.",
            "sa_to_bbox_volume_ratio": "Surface area to bounding box volume ratio.",
            "num_vertices": "Number of vertices in the mesh.",
            "num_faces": "Number of faces in the mesh.",
            "mean_curvature": "Mean curvature of the mesh.",
            "std_curvature": "Standard deviation of the curvature.",
            "mean_edge_length": "Mean edge length of the mesh.",
            "std_edge_length": "Standard deviation of the edge lengths.",
            "mean_face_area": "Mean face area.",
            "std_face_area": "Standard deviation of the face areas.",
            "axis_ratio": "Ratio of major axis to minor axis in the XY-plane.",
            "median_depth": "Median depth (Z) of the mesh."
        }


def load_meshes_from_folders(datasets_folder):
    """Load meshes from a folder hierarchy, ignoring any 2D images."""
    meshes = []
    subfolder_names = []

    for subfolder_name in os.listdir(datasets_folder):
        subfolder_path = os.path.join(datasets_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            ply_file = None

            # Look for 'mask.ply' or similar
            for file_name in os.listdir(subfolder_path):
                if file_name == 'mask.ply':
                    ply_file = os.path.join(subfolder_path, file_name)
                    break

            if ply_file:
                mesh = trimesh.load(ply_file)
                meshes.append(mesh)
                subfolder_names.append(subfolder_name)
            else:
                print(f"Skipping {subfolder_name} because no 'mask.ply' found.")

    return meshes, subfolder_names


def normalize_mesh(mesh, only_scale=False):
    """Normalize the mesh for translation, rotation, and scale invariance."""
    mesh_copy = mesh.copy()

    if not only_scale:
        # Translate centroid to origin
        centroid = mesh_copy.vertices.mean(axis=0)
        mesh_copy.vertices -= centroid

        # Align principal axes with PCA
        pca = PCA(n_components=3)
        pca.fit(mesh_copy.vertices)
        mesh_copy.vertices = np.dot(mesh_copy.vertices, pca.components_.T)

    # Scale to fit within a unit sphere
    scale = np.linalg.norm(mesh_copy.vertices, axis=1).max()
    if scale != 0:
        mesh_copy.vertices /= scale

    return mesh_copy


def compute_laplacian_curvature_vectorized(mesh):
    """Compute mean curvature using a Laplace-Beltrami approach with a sparse adjacency matrix."""
    vertex_count = len(mesh.vertices)
    vertex_indices = np.arange(vertex_count)

    # Build adjacency
    data = []
    rows = []
    cols = []
    for vi in vertex_indices:
        neighbors = mesh.vertex_neighbors[vi]
        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            data.extend([1]*num_neighbors)
            rows.extend([vi]*num_neighbors)
            cols.extend(neighbors)

    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))

    # Degree matrix
    degrees = adjacency_matrix.sum(axis=1).A1
    degree_matrix = csr_matrix((degrees, (vertex_indices, vertex_indices)), shape=(vertex_count, vertex_count))

    # Laplacian
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Laplacian * vertices
    laplacian = laplacian_matrix.dot(mesh.vertices)

    # Normalize by degree
    degrees[degrees == 0] = 1.0  # avoid division by zero
    laplacian /= degrees[:, np.newaxis]

    curvature = np.linalg.norm(laplacian, axis=1)
    curvature[np.isnan(curvature)] = 0
    return curvature


def compute_main_axis_ratio(mesh):
    """
    Compute ratio of major axis length to minor axis length in the XY-plane after rotation alignment.
    """
    vertices = np.asarray(mesh.vertices)
    rotation_matrix = find_rotation_matrix(vertices)
    aligned_vertices = np.dot(vertices, rotation_matrix.T)

    xy_projection = aligned_vertices[:, :2]
    pca_2d = PCA(n_components=2)
    pca_2d.fit(xy_projection)
    transformed = pca_2d.transform(xy_projection)

    major_axis_length = np.max(transformed[:, 0]) - np.min(transformed[:, 0])
    minor_axis_length = np.max(transformed[:, 1]) - np.min(transformed[:, 1])
    if minor_axis_length == 0:
        return 0
    axis_ratio = major_axis_length / minor_axis_length
    return axis_ratio


def compute_median_depth(mesh):
    """Compute the median Z (depth) after best-fit alignment."""
    vertices = np.asarray(mesh.vertices)
    rotation_matrix = find_rotation_matrix(vertices)
    aligned_vertices = np.dot(vertices, rotation_matrix.T)
    return np.median(aligned_vertices[:, 2])


def extract_geometric_features(meshes):
    """
    Extract a set of geometric features from each mesh.
    Returns (feature_matrix, list_of_dicts, feature_names).
    """
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
        'std_face_area',
        'axis_ratio',
        'median_depth'
    ]

    for mesh in meshes:
        # Create scaled and fully-normalized versions if needed
        scaled_mesh = normalize_mesh(mesh, only_scale=True)
        normalized_mesh = normalize_mesh(mesh, only_scale=False)

        # Basic stats
        surface_area = scaled_mesh.area
        bbox = scaled_mesh.bounds  # shape (2,3)
        bbox_size = bbox[1] - bbox[0]
        volume_bbox = np.prod(bbox_size) if np.prod(bbox_size) != 0 else 1
        sa_to_bbox_vol = surface_area / volume_bbox

        num_vertices = len(scaled_mesh.vertices)
        num_faces = len(scaled_mesh.faces)

        # Curvature from scaled_mesh
        curvature = compute_laplacian_curvature_vectorized(scaled_mesh)
        mean_curv = np.mean(curvature)
        std_curv = np.std(curvature)

        # Edge lengths
        edges = scaled_mesh.edges_unique_length
        mean_edge_len = np.mean(edges) if len(edges) else 0
        std_edge_len = np.std(edges) if len(edges) else 0

        # Face areas
        face_areas = scaled_mesh.area_faces
        mean_fa = np.mean(face_areas) if len(face_areas) else 0
        std_fa = np.std(face_areas) if len(face_areas) else 0

        # Axis ratio & median depth from fully normalized mesh
        axis_ratio = compute_main_axis_ratio(normalized_mesh)
        median_depth = compute_median_depth(normalized_mesh)

        feature_vector = np.array([
            surface_area,
            sa_to_bbox_vol,
            num_vertices,
            num_faces,
            mean_curv,
            std_curv,
            mean_edge_len,
            std_edge_len,
            mean_fa,
            std_fa,
            axis_ratio,
            median_depth
        ])

        features.append(feature_vector)
        feature_data.append({
            'mesh': mesh,
            'surface_area': surface_area,
            'sa_to_bbox_volume_ratio': sa_to_bbox_vol,
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'curvature': curvature,
            'mean_curvature': mean_curv,
            'std_curvature': std_curv,
            'edge_lengths': edges,
            'mean_edge_length': mean_edge_len,
            'std_edge_length': std_edge_len,
            'face_areas': face_areas,
            'mean_face_area': mean_fa,
            'std_face_area': std_fa,
            'axis_ratio': axis_ratio,
            'median_depth': median_depth
        })

    return np.vstack(features), feature_data, feature_names
