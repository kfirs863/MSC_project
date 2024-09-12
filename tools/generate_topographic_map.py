import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.utils import preprocess_mesh


def normalize_mesh(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize the mesh by centering it around the origin and scaling it to fit in a unit cube.

    Args:
        vertices: Numpy array of shape (N, 3) representing the vertices of the mesh.

    Returns:
        Normalized vertices.
    """
    # Center the mesh
    centroid = np.mean(vertices, axis=0)
    vertices -= centroid

    # Scale the mesh to fit in a unit bounding box
    max_range = np.max(np.abs(vertices))  # Max range across all axes
    vertices /= max_range

    return vertices


def generate_topographic_map(folder_path: str, num_contours: int = 1, num_iterations: int = 4, mode: str = 'equal',
                             fixed_depths: list = None, normalize: bool = False) -> list:
    """
    Generate a topographic map from a 3D mesh in a .ply file found in the given folder.

    Args:
        folder_path: Path to the folder containing the .ply file
        num_contours: Number of contour lines to generate (only used in 'equal' mode)
        num_iterations: Number of iterations for mesh subdivision
        mode: 'equal' for evenly spaced contours, 'fixed' for user-defined depth levels
        fixed_depths: List of fixed Z values for contour generation (only used in 'fixed' mode)
        normalize: Whether to normalize the mesh (center and scale it to fit in a unit cube)

    Returns:
        List of paths to saved images of the contour lines
    """

    # Find the .ply file in the given folder
    ply_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.ply'):
            ply_file = os.path.join(folder_path, file)
            break

    if not ply_file:
        raise FileNotFoundError("No .ply file found in the specified folder.")

    # Load the mesh from the .ply file
    mesh,_ = preprocess_mesh(ply_file)

    # Smooth and subdivide the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh = mesh.subdivide_midpoint(number_of_iterations=num_iterations)

    # Check for non-manifold geometry and remove issues
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Compute normals (optional, for visualization)
    mesh.compute_vertex_normals()

    # Extract vertices
    vertices = np.asarray(mesh.vertices)

    # Normalize the mesh if requested
    if normalize:
        vertices = normalize_mesh(vertices)

    # Translate the mesh so the maximum Z value is at zero
    z_max = np.max(vertices[:, 2])
    vertices[:, 2] -= z_max

    # Calculate the new min and max Z values after translation
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])

    # Generate Z levels based on mode
    if mode == 'equal':
        # Generate evenly spaced Z levels across the Z-axis range
        z_levels = np.linspace(z_min, z_max, num_contours)
    elif mode == 'fixed':
        if not fixed_depths:
            raise ValueError("fixed_depths must be provided in 'fixed' mode.")
        # Use provided fixed depth values
        z_levels = np.array(fixed_depths)
    else:
        raise ValueError("Invalid mode. Use 'equal' or 'fixed'.")

    # List to hold the paths of saved images
    saved_image_paths = []

    # Increased threshold to capture more points near the target Z level
    threshold = 0.001

    # Plot each contour and save to a file
    for i, z in enumerate(z_levels):
        # Find points close to the current Z level
        points_at_z = vertices[np.abs(vertices[:, 2] - z) < threshold]

        if len(points_at_z) > 0:
            plt.figure()
            # Increased marker size for better visibility
            plt.scatter(points_at_z[:, 0], points_at_z[:, 1], s=1)
            plt.gca().set_aspect('equal', adjustable='box')

            # Remove axes for a cleaner visualization
            plt.axis('off')
            image_path = os.path.join(folder_path, f'contour_{i}.png')
            plt.savefig(image_path)
            plt.close()
            saved_image_paths.append(image_path)

    return saved_image_paths

if __name__ == '__main__':
    folder_path ='/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets/S01_mask_1' # Specify the folder path containing the .ply file
    num_contours = 3  # Specify the number of contour lines
    generate_topographic_map(folder_path=folder_path, num_contours=10, num_iterations=4, mode='equal', normalize=True)
