import os
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

def find_rotation_matrix(vertices):
    """Find the rotation matrix to align the dominant plane of the point cloud with the XY plane."""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    normal = pca.components_[2]  # The normal to the plane is the last principal component
    z_axis = np.array([0, 0, -1])

    # Compute the rotation matrix to align the normal with the z-axis
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    s = np.linalg.norm(v)

    if s == 0:  # If the normal is already aligned with the z-axis
        rotation_matrix = np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def rotate_mesh(mesh, rotation_matrix):
    """Apply a rotation to the mesh."""
    vertices = np.asarray(mesh.vertices)
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    return mesh

def preprocess_mesh(obj_path):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)

    if not mesh.has_triangles():
        raise ValueError("The mesh does not contain any triangles.")

    # Ensure the mesh has textures
    if not mesh.has_textures():
        raise ValueError("The mesh does not contain textures.")

    # Ensure the mesh has vertex normals
    mesh.compute_vertex_normals()

    # Convert Open3D mesh to NumPy array
    vertices = np.asarray(mesh.vertices)

    # Find the rotation matrix
    rotation_matrix = find_rotation_matrix(vertices)

    # Rotate the mesh
    rotated_mesh = rotate_mesh(mesh, rotation_matrix)

    return rotated_mesh, rotation_matrix
