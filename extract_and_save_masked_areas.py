import open3d as o3d
import numpy as np
import os

def extract_and_save_masked_areas(input_ply_path, output_dir):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_ply_path)

    # Convert point cloud to numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Identify unique colors (ignore black [0, 0, 0])
    unique_colors = np.unique(colors, axis=0)
    unique_colors = unique_colors[np.any(unique_colors != [0, 0, 0], axis=1)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each unique color
    for i, color in enumerate(unique_colors):
        # Filter points matching the current color
        color_mask = np.all(colors == color, axis=1)
        filtered_points = points[color_mask]
        filtered_colors = colors[color_mask]

        # Create a new point cloud with the filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Save the filtered point cloud to a new file
        output_ply_path = os.path.join(output_dir, f"masked_area_{i}.ply")
        o3d.io.write_point_cloud(output_ply_path, filtered_pcd)

        print(f"Saved masked area {i} with color {color} to {output_ply_path}")

    print("Finished extracting and saving all masked areas.")

# Function do display with open3d all the  masks in a folder
def display_masked_areas(output_dir):
    # List all the files in the output directory
    files = os.listdir(output_dir)

    # Load each file and display it
    for i, file in enumerate(files):
        if file.endswith(".ply"):
            pcd = o3d.io.read_point_cloud(os.path.join(output_dir, file))
            o3d.visualization.draw_geometries([pcd], window_name=f"Masked Area {i}")

    print(f"Displayed {len(files)} masked areas.")

if __name__ == '__main__':
    display_masked_areas("./images/splited_staircase_left_masked_point_clouds")

    # # Example usage:
    input_ply_path = "notebook/images/staircase_left_masked_point_cloud.ply"
    output_dir = "./images/splited_staircase_left_masked_point_clouds"
    extract_and_save_masked_areas(input_ply_path, output_dir)

