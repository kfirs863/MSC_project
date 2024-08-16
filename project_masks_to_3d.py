import open3d as o3d
import numpy as np
import json

from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import find_rotation_matrix, center_data


def project_2d_to_3d(u, v, depth, intrinsics, extrinsics):
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    z = depth[v, u]
    if z < 1e-6:  # Skip points with near-zero depth
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_3d = np.array([x, y, z, 1.0])

    # Apply the extrinsics to get the 3D point in world coordinates
    point_3d_world = np.dot(extrinsics, point_3d)

    return point_3d_world[:3]

def center_and_align(vertices):
    """Center and align the vertices using PCA to align the x, y, and z axes."""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    aligned_vertices = pca.transform(vertices)
    return aligned_vertices, pca

def project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path):
    # Load the masks and mask colors
    best_masks = np.load(masks_path)
    mask_colors = np.load(colors_path)

    # Load the saved camera parameters
    with open(params_path, 'r') as f:
        params = json.load(f)

    camera_intrinsics = params['camera_intrinsics']
    rotation_matrix = np.array(params['rotation_matrix'])
    extrinsic = np.array(params['extrinsic'])

    # Load the depth image
    depth_np = np.load(depth_path)

    # Combine all masks to create a full mask
    combined_mask = np.zeros(best_masks[0].shape, dtype=bool)
    for mask in best_masks:
        combined_mask |= mask

    # Create the black mask for pixels not belonging to any mask
    black_mask = ~combined_mask

    # Process the original masks
    points_3d = []
    colors = []

    for mask_idx in tqdm(range(best_masks.shape[0]), desc="Processing masks"):
        mask = best_masks[mask_idx]
        color = mask_colors[mask_idx]

        # Get the pixel coordinates of the mask
        mask_indices = np.argwhere(mask)

        for v, u in mask_indices:
            point_3d = project_2d_to_3d(u, v, depth_np, camera_intrinsics, extrinsic)
            if point_3d is not None:
                points_3d.append(point_3d)
                colors.append(color)

    # Process the black mask
    black_mask_indices = np.argwhere(black_mask)

    for v, u in tqdm(black_mask_indices, desc="Processing black mask"):
        point_3d = project_2d_to_3d(u, v, depth_np, camera_intrinsics, extrinsic)
        if point_3d is not None:
            points_3d.append(point_3d)
            colors.append([0, 0, 0])  # Color black

    points_3d = np.array(points_3d)
    colors = np.array(colors) / 255.0  # Normalize colors to range [0, 1] for Open3D

    # Create a point cloud for the mask and black mask
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(points_3d)
    combined_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Load the mesh from the obj_path
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices)

    # Center both the mesh and the point cloud
    mesh_vertices_centered, mesh_center = center_data(mesh_vertices)
    points_3d_centered, pcd_center = center_data(points_3d)

    # Apply the centering to the mesh
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_centered)

    # Find the rotation matrix for the mesh vertices
    rotation_matrix = find_rotation_matrix(mesh_vertices_centered)

    # Apply the rotation matrix to the mesh
    mesh_vertices_rotated = np.dot(mesh_vertices_centered, rotation_matrix.T)
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_rotated)

    # Also rotate the point cloud points using the same matrix
    points_3d_rotated = np.dot(points_3d_centered, rotation_matrix.T)
    combined_pcd.points = o3d.utility.Vector3dVector(points_3d_rotated)

    # Initialize the mesh colors with the original mesh vertex colors
    if mesh.has_vertex_colors():
        mesh_colors = np.asarray(mesh.vertex_colors)
    else:
        mesh_colors = np.ones_like(mesh_vertices_rotated)  # Default to white if no colors are available

    # Create a KDTree for fast nearest-neighbor lookup
    pcd_tree = o3d.geometry.KDTreeFlann(combined_pcd)

    for i, vertex in enumerate(mesh_vertices_rotated):
        # Find the nearest point in the point cloud
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        nearest_color = np.asarray(combined_pcd.colors)[idx[0]]

        # Assign the color of the nearest point to the mesh vertex only if the color is not black
        if not np.all(nearest_color == [0, 0, 0]):
            mesh_colors[i] = nearest_color

    # Apply the colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    # Save the colored mesh
    colored_mesh_path = params_path.replace("_params.json", "_colored_mesh.obj")
    o3d.io.write_triangle_mesh(colored_mesh_path, mesh)

    # Visualize the colored mesh
    o3d.visualization.draw_geometries([mesh])

    return colored_mesh_path


if __name__ == '__main__':
    # Usage
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/S01/S01.obj'
    params_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_params.json'
    masks_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_ortho_masks.npy'
    colors_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_ortho_colors.npy'
    depth_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_depth.npy'

    # Generate the colored mesh
    colored_mesh_path = project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path)