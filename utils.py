import json
import os
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import cv2

def find_rotation_matrix(vertices):
    """Find the rotation matrix to align the dominant plane of the point cloud with the XY plane."""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    normal = pca.components_[2]  # The normal to the plane is the last principal component
    z_axis = np.array([0, 0, -1])  # Z-axis is pointing to the opposite direction of the PCA normal

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
    """Load and preprocess the mesh by rotating it to align with the XY plane."""
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    except Exception as e:
        raise ValueError(f"Error reading the mesh: {e}")

    if not mesh.has_triangles():
        raise ValueError("The mesh does not contain any triangles.")

    if not mesh.has_textures():
        raise ValueError("The mesh does not contain textures.")

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    rotation_matrix = find_rotation_matrix(vertices)
    rotated_mesh = rotate_mesh(mesh, rotation_matrix)

    return rotated_mesh, rotation_matrix

def save_image_and_params(image_np, depth_np, obj_stem, rotation_matrix, camera_intrinsics, extrinsic):
    """Save the captured image and camera parameters."""
    os.makedirs("./images", exist_ok=True)
    output_image_path = f"./images/{obj_stem}_ortho.png"
    output_depth_path = f"./images/{obj_stem}_depth.npy"
    cv2.imwrite(output_image_path, image_np)
    np.save(output_depth_path, depth_np)

    output_params_path = f"./images/{obj_stem}_params.json"
    params = {
        'camera_intrinsics': camera_intrinsics,
        'rotation_matrix': rotation_matrix.tolist(),
        'extrinsic': extrinsic.tolist()  # Add extrinsic parameters
    }
    with open(output_params_path, 'w') as f:
        json.dump(params, f, indent=4)

    return output_image_path, output_depth_path, output_params_path
