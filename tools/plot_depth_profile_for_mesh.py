from pathlib import Path

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import os


def preprocess_and_align_mesh(mesh):
    # Recompute normals to ensure they are consistent
    mesh.compute_vertex_normals()
    return mesh


def plot_depth_profile_for_mesh(folder_path, num_sections=50, smoothing_sigma=20):

    # Find the .ply file in the given folder
    ply_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.ply'):
            ply_file = os.path.join(folder_path, file)
            break

    if not ply_file:
        raise FileNotFoundError("No .ply file found in the specified folder.")

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(ply_file)

    if mesh.is_empty():
        raise ValueError(f"Mesh {ply_file} is empty or could not be loaded.")

    # Preprocess and align the mesh
    mesh = preprocess_and_align_mesh(mesh)


    # Subdivide the mesh to increase the number of points
    mesh = mesh.subdivide_midpoint(
        number_of_iterations=4)  # Adjust the number of iterations for finer mesh


    # Convert mesh vertices to a numpy array
    vertices = np.asarray(mesh.vertices)

    # Extract the X, Y, Z coordinates
    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]

    # Set the highest point as zero by subtracting the maximum Z value from all Z values
    max_z = np.max(Z)
    depth = Z - max_z

    # Shift X values to start from zero
    X_shifted = X - np.min(X)
    max_x_shifted = np.max(X_shifted)

    # Determine the range of Y
    min_y, max_y = np.min(Y), np.max(Y)
    mid_y = (min_y + max_y) / 2

    # Define horizontal cross-sections starting from the middle
    if num_sections == 1:
        y_sections = [mid_y]
    else:
        half_sections = num_sections // 2
        y_sections = np.linspace(mid_y, max_y, half_sections, endpoint=False)
        y_sections = np.concatenate([y_sections, np.linspace(mid_y, min_y, half_sections, endpoint=True)])

    plt.figure(figsize=(10, 6))

    # Use a colormap to assign different colors to each line
    colormap = cm.get_cmap('viridis', num_sections)

    # Define a common X-range for all sections
    common_x_range = np.linspace(0, max_x_shifted, 500)

    # For each horizontal section, plot the depth profile
    for i, y_val in enumerate(y_sections):
        # Find points in the current horizontal section
        section_mask = np.abs(Y - y_val) < (max_y - min_y) / (2 * num_sections)
        section_x = X_shifted[section_mask]
        section_depth = depth[section_mask]

        # Skip this section if it doesn't have enough data points
        if len(section_x) < 2:
            continue

        # Sort by X to plot a profile
        sorted_indices = np.argsort(section_x)
        section_x_sorted = section_x[sorted_indices]
        section_depth_sorted = section_depth[sorted_indices]

        # Apply smoothing to the depth values
        section_depth_smoothed = gaussian_filter1d(section_depth_sorted, sigma=smoothing_sigma)

        # Interpolate across the common X-range, filling missing data with zero (default depth)
        interpolation_func = interp1d(section_x_sorted, section_depth_smoothed, kind='linear', bounds_error=False,
                                      fill_value=0)
        interpolated_depth = interpolation_func(common_x_range)

        # Plot the depth profile with a unique color
        plt.plot(common_x_range, interpolated_depth, color=colormap(i), label=f'Section {i + 1}')

    # plt.xlabel('X (mm)')
    # plt.ylabel('Depth from Surface (mm)')
    # plt.title('Depth Profile in Horizontal Cross-Section')
    # # plt.legend()
    # plt.grid(True)

    # Save the depth profile plot
    depth_profile_path = Path(folder_path, 'depth_profile.png')
    plt.savefig(depth_profile_path)
    plt.close()

    # Create a 2D visualization of the mesh with the cross-section lines
    plt.figure(figsize=(8, 8))
    plt.scatter(X_shifted, Y, c=Z, cmap='gray', s=1, alpha=0.6)  # Mesh scatter plot

    # Plot the cross-section lines
    for i, y_val in enumerate(y_sections):
        plt.axhline(y=y_val, color=colormap(i), linestyle='-', linewidth=1, label=f'Section {i + 1}')

    # plt.xlabel('X (mm)')
    # plt.ylabel('Y (mm)')
    # plt.title('2D View of Mesh with Cross-Section Lines')
    # # plt.legend(loc='upper right')
    # plt.grid(True)

    # Save the 2D view plot
    view_2d_path = Path(folder_path, 'view_2d.png')
    plt.savefig(view_2d_path)
    plt.close()

    return depth_profile_path, view_2d_path


if __name__ == '__main__':
    # Example usage
    ply_file_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/Crosses on Staircase left/masked_areas/masked_area_7.ply'
    depth_profile_path, view_2d_path = plot_depth_profile_for_mesh(ply_file_path, num_sections=50)
    print(f"Depth profile saved to: {depth_profile_path}")
    print(f"2D view saved to: {view_2d_path}")