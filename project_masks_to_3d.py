import open3d as o3d
import numpy as np
import cv2
import json

from utils import preprocess_mesh


def project_masks_to_point_cloud(obj_path, json_path, masks, mask_colors):
    rotated_mesh, rotation_matrix = preprocess_mesh(obj_path)
    # Load the 3D model's point cloud
    rotated_mesh.compute_vertex_normals()
    point_cloud = o3d.utility.Vector3dVector(rotated_mesh.vertices)

    # Load camera intrinsic parameters and the rotation matrix from the JSON file
    with open(json_path, 'r') as f:
        params = json.load(f)
    camera_intrinsics = params['camera_intrinsics']
    rotation_matrix = np.array(params['rotation_matrix'])

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Transform 3D points using the inverse of the rotation matrix
    rotated_vertices = np.asarray(point_cloud.points)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    transformed_vertices = np.dot(rotated_vertices, inverse_rotation_matrix.T)

    # Project 3D points to 2D using intrinsic parameters
    points_2d = np.zeros((transformed_vertices.shape[0], 2))
    points_2d[:, 0] = fx * (transformed_vertices[:, 0] / -transformed_vertices[:, 2]) + cx
    points_2d[:, 1] = fy * (transformed_vertices[:, 1] / -transformed_vertices[:, 2]) + cy

    # Create an array to store colors for each 3D point
    point_colors = np.zeros((transformed_vertices.shape[0], 3), dtype=np.uint8)

    # Iterate through the masks and apply colors to the points
    for mask, color in zip(masks, mask_colors):
        for i, (x, y) in enumerate(points_2d):
            x, y = int(x), int(y)
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x] > 0:  # If the mask is present at this point
                    point_colors[i] = color

    # Apply the colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors / 255.0)

    # Save or visualize the colored point cloud
    o3d.io.write_point_cloud("colored_point_cloud.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])

    return point_cloud


if __name__ == '__main__':
    obj_path = 'path/to/your/object.obj'
    json_path = 'path/to/your/params.json'

    # Load the masks (assuming masks is a list of 2D numpy arrays)
    # masks = [cv2.imread('path/to/mask1.png', cv2.IMREAD_GRAYSCALE), ...]

    # Define the mask colors (assuming mask_colors is a list of RGB tuples)
    # mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), ...]

    # Example usage
    point_cloud = project_masks_to_point_cloud(obj_path, json_path, masks, mask_colors)
