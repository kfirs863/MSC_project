import open3d as o3d
import numpy as np
import json

from tqdm import tqdm

from tools.utils import preprocess_mesh

def project_2d_to_3d(u, v, depth, intrinsics, extrinsic):
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    z = depth[v, u]
    if not z:  # Skip points with zero depth
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_3d = np.array([x, y, z, 1.0])

    # Apply the extrinsic to get the 3D point in world coordinates
    point_3d_world = np.dot(np.linalg.inv(extrinsic), point_3d)

    return point_3d_world[:3]

def project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path):
    # Load the masks and mask colors
    best_masks = np.load(masks_path)
    mask_colors = np.load(colors_path)

    # Load the saved camera parameters
    with open(params_path, 'r') as f:
        params = json.load(f)

    camera_intrinsics = params['camera_intrinsics']
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

    for mask_idx in tqdm(range(best_masks.shape[0]), desc="Processing masks",unit="mask"):
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

    for v, u in tqdm(black_mask_indices, desc="Processing non-masked areas",unit="pixel"):
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
    mesh, rotation_matrix_mesh = preprocess_mesh(obj_path)


    # Visualize the combined point cloud and the mesh
    # o3d.visualization.draw_geometries([combined_pcd, mesh])


    # Initialize the mesh colors with the original mesh vertex colors
    mesh_colors = np.ones_like(mesh.vertices)  # Default to white if no colors are available

    # Create a KDTree for fast nearest-neighbor lookup
    pcd_tree = o3d.geometry.KDTreeFlann(combined_pcd)

    for i, vertex in tqdm(enumerate(mesh.vertices), desc="Assigning colors to mesh vertices",unit="vertex"):
        # Find the nearest point in the point cloud
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        nearest_color = np.asarray(combined_pcd.colors)[idx[0]]

        # Assign the color of the nearest point to the mesh vertex only if the color is not black
        if not np.all(nearest_color == [0, 0, 0]):
            mesh_colors[i] = nearest_color

    # Apply the colors to the mesh
    mesh = mesh.filter_sharpen(number_of_iterations=1, strength=0.01)
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    mesh.compute_vertex_normals()

    # Optional: Recompute vertex normals if needed
    # mesh.compute_vertex_normals()

    # Save the colored mesh
    colored_mesh_path = params_path.replace("_params.json", "_colored_mesh.ply")
    o3d.io.write_triangle_mesh(colored_mesh_path, mesh)

    # Visualize the colored mesh
    # o3d.visualization.draw_geometries([mesh], window_name="Colored Mesh",mesh_show_back_face=True)

    return colored_mesh_path


if __name__ == '__main__':
    # Usage
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/Crosses on Staircase left/staircase_left.obj'
    params_path = '/notebook/images/staircase_left_params.json'
    masks_path = '/notebook/images/staircase_left_ortho_masks.npy'
    colors_path = '/notebook/images/staircase_left_ortho_colors.npy'
    depth_path = '/notebook/images/staircase_left_depth.npy'

    # Generate the colored mesh
    colored_mesh_path = project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path)
