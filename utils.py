import json
import os
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import cv2

import torch
from PIL import Image, ImageEnhance
import numpy as np
from RealESRGAN import RealESRGAN



def center_data(vertices):
    """Center the vertices of the point cloud or mesh to the origin."""
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    return centered_vertices, center


def find_rotation_matrix(vertices: np.array, flip_z: bool = True):
    """Find the rotation matrix to align the dominant plane of the point cloud with the XY plane."""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    normal = pca.components_[2]  # The normal to the plane is the last principal component
    z_axis = np.array([0, 0, -1]) if flip_z else np.array([0, 0, 1])

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

def rotate_yaw(vertices, angle_degrees):
    """Rotate the vertices around the Z axis (yaw rotation)."""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return np.dot(vertices, rotation_matrix.T)

def rotate_mesh(mesh, rotation_matrix):
    """Apply a rotation to the mesh."""
    vertices = np.asarray(mesh.vertices)
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    return mesh

def preprocess_mesh(obj_path, flip_z=True, yaw_angle_degrees=0):
    """Load and preprocess the mesh by rotating it to align with the Z axis and then applying a yaw rotation."""
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
    rotation_matrix = find_rotation_matrix(vertices, flip_z)
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Apply yaw rotation
    rotated_vertices = rotate_yaw(rotated_vertices, yaw_angle_degrees)
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


def color_mesh_by_vertex_normals(mesh):
    """Colors the mesh by its vertex normals."""
    mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Paint with a gray color
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1) / 2  # Normalize normals to the range [0, 1]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

def super_resolution(input_image, factor=2):
    device = torch.device('cpu')

    model = RealESRGAN(device, scale=factor)
    model.load_weights(f'weights/RealESRGAN_x{factor}.pth', download=False)

    # Convert numpy array to PIL Image
    image = Image.fromarray(input_image)

    # Apply super-resolution
    sr_image = model.predict(image)

    # Convert back to numpy array
    sr_image_np = np.array(sr_image)

    return sr_image_np

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
