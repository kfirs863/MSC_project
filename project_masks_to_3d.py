import open3d as o3d
import numpy as np
import json

from tqdm import tqdm


def project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path):
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

    # Save the combined point cloud
    combined_point_cloud_path = "combined_colored_points.ply"
    o3d.io.write_point_cloud(combined_point_cloud_path, combined_pcd)

    # Visualize the combined point cloud
    o3d.visualization.draw_geometries([combined_pcd])

    return combined_point_cloud_path
if __name__ == '__main__':
    # Usage
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/S01/S01.obj'
    params_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_params.json'
    masks_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_ortho_masks.npy'
    colors_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_ortho_colors.npy'
    depth_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/S01_depth.npy'

    # Usage
    project_masks_to_mesh(obj_path, masks_path, colors_path, params_path, depth_path)

    # Visualize or save the result
    # o3d.visualization.draw_geometries([colored_mesh])
    # # o3d.io.write_triangle_mesh("colored_mesh.obj", colored_mesh)