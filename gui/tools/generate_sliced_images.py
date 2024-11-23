import os

import numpy as np
from matplotlib import pyplot as plt

from tools.utils import preprocess_mesh


def generate_sliced_images(ply_file, num_slices=5, slice_thickness=0.01, trim_percent=0.1):
    """
    Generate images of the mesh at different Z levels (slices).

    Args:
        ply_file (str): Path to the .ply file.
        num_slices (int): Number of slices to generate.
        slice_thickness (float): Thickness of each slice as a proportion of the Z range.
        trim_percent (float): Percentage of Z range to trim from both ends (between 0 and 0.5).

    Returns:
        saved_image_paths (list): List of paths to the saved slice images.
    """
    # Load and preprocess the mesh
    mesh, _ = preprocess_mesh(ply_file)

    # Subdivide the mesh
    mesh = mesh.subdivide_midpoint(number_of_iterations=4)

    vertices = np.asarray(mesh.vertices)
    X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Normalize the mesh to [0, 1]
    X_min, X_max = X.min(), X.max()
    Y_min, Y_max = Y.min(), Y.max()
    Z_min, Z_max = Z.min(), Z.max()

    X_normalized = (X - X_min) / (X_max - X_min)
    Y_normalized = (Y - Y_min) / (Y_max - Y_min)
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)

    # Trim the Z range based on the specified percentage
    trim_percent = max(0, min(trim_percent, 0.5))
    lower_percentile = trim_percent * 100
    upper_percentile = 100 - trim_percent * 100
    Z_values = Z_normalized
    Z_min_adjusted = np.percentile(Z_values, lower_percentile)
    Z_max_adjusted = np.percentile(Z_values, upper_percentile)

    # Generate Z levels (slice centers) within the adjusted Z range
    z_levels = np.linspace(Z_min_adjusted, Z_max_adjusted, num_slices)

    # Define the actual slice thickness in Z units
    z_range = Z_max_adjusted - Z_min_adjusted
    slice_thickness = slice_thickness * z_range  # Convert proportion to actual value

    saved_image_paths = []
    output_dir = os.path.dirname(ply_file)

    # For each Z level, extract the slice and save the image
    for i, z in enumerate(z_levels):
        # Find points within the slice thickness around the current Z level
        slice_mask = np.abs(Z_normalized - z) < (slice_thickness / 2)
        X_slice = X_normalized[slice_mask]
        Y_slice = Y_normalized[slice_mask]

        # Skip if not enough points in the slice
        if len(X_slice) < 10:
            continue

        # Plot the slice
        plt.figure(figsize=(6, 6))
        plt.scatter(X_slice, Y_slice, s=1, color='black')
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout(pad=0)

        # Save the image
        image_path = os.path.join(output_dir, f'slice_{i}.png')
        plt.savefig(image_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        saved_image_paths.append(image_path)

    return saved_image_paths

# Example usage
if __name__ == '__main__':

    def find_ply_files(root_folder):
        ply_files = []

        for dirpath, dirnames, filenames in os.walk(root_folder):
            for file in filenames:
                if file.endswith('.ply'):
                    ply_files.append(os.path.join(dirpath, file))

        return ply_files


    # Usage example:
    root_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets'
    ply_files = find_ply_files(root_path)

    for ply_file in ply_files:
        print(ply_file)
        saved_images = generate_sliced_images(ply_file, num_slices=4, slice_thickness=0.05, trim_percent=0.1)
        print("Saved slice images:")
        for path in saved_images:
            print(path)

