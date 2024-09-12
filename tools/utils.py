import json
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageEnhance
from sklearn.decomposition import PCA


def find_rotation_matrix(vertices: np.array):
    """Find the rotation matrix to align the dominant plane of the point cloud with the XY plane."""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    normal = pca.components_[2]  # The normal to the plane is the last principal component
    z_axis = np.array([0, 0, 1])

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
    """Load and preprocess the mesh by rotating it to align with the Z axis and then applying a yaw rotation."""
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    except Exception as e:
        raise ValueError(f"Error reading the mesh: {e}")

    if not mesh.has_triangles():
        raise ValueError("The mesh does not contain any triangles.")

    # if not mesh.has_textures():
    #     raise ValueError("The mesh does not contain textures.")

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    rotation_matrix = find_rotation_matrix(vertices)
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)

    return mesh, rotation_matrix

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

def enhance_image(image_np, sharpness=4, contrast=1.3, blur=3):
    """Enhance image sharpness, contrast, and blur.

    Args:
        image_np (np.array): Input image as a NumPy array.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.

    Returns:
        np.array: Enhanced image as a NumPy array.
    """

    # Convert the image to PIL Image
    pil_img = Image.fromarray(image_np)

    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)

    # Convert back to OpenCV image (numpy array)
    img_enhanced = np.array(img_enhanced)

    # Apply a small amount of Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)

    return img_enhanced

def display_masked_areas(output_dir):
    # List all the files in the output directory
    files = os.listdir(output_dir)

    # Load each file and display it
    for i, file in enumerate(files):
        if file.endswith(".ply"):
            # Load the mesh instead of point cloud
            masked_mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, file))
            if masked_mesh.is_empty():
                print(f"Skipped empty mesh file: {file}")
                continue
            masked_mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([masked_mesh], window_name=f"Masked Area {i}",mesh_show_back_face=True)

    print(f"Displayed {len(files)} masked areas.")

def extract_xyz_from_ply(ply_file,number_of_iterations=4):
    """Extract the XYZ coordinates from a PLY file."""
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(ply_file)

    if mesh.is_empty():
        raise ValueError(f"Mesh {ply_file} is empty or could not be loaded.")

    # Preprocess and align the mesh
    mesh.compute_vertex_normals()


    # Subdivide the mesh to increase the number of points
    mesh = mesh.subdivide_midpoint(
        number_of_iterations=number_of_iterations)  # Adjust the number of iterations for finer mesh

    # Convert mesh vertices to a numpy array
    vertices = np.asarray(mesh.vertices)

    # Extract the X, Y, Z coordinates
    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]

    return X, Y, Z
