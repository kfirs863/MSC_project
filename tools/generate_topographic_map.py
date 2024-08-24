import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def generate_topographic_map(folder_path: str, num_contours: int = 1, num_iterations: int = 4) -> list:
    """
    Generate a topographic map from a 3D mesh in a .ply file found in the given folder.
    Args:
        folder_path: Path to the folder containing the .ply file
        num_contours: Number of contour lines to generate
        num_iterations: Number of iterations for mesh subdivision

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
    mesh = o3d.io.read_triangle_mesh(ply_file)
    mesh.compute_vertex_normals()

    # Subdivide the mesh to increase the number of points
    mesh = mesh.subdivide_midpoint(number_of_iterations=num_iterations)

    # Extract vertices
    vertices = np.asarray(mesh.vertices)

    # Translate the mesh so the maximum Z value is at zero
    z_max = np.max(vertices[:, 2])
    vertices[:, 2] -= z_max

    # Calculate the new min and max Z values after translation
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])

    # Generate evenly spaced Z levels across the Z-axis range
    z_levels = np.linspace(z_min, z_max, num_contours)

    # List to hold the paths of saved images
    saved_image_paths = []

    # Plot each contour and save to a file
    for i, z in enumerate(z_levels):
        # Find points close to the current Z level
        threshold = 0.001
        points_at_z = vertices[np.abs(vertices[:, 2] - z) < threshold]

        if len(points_at_z) > 0:
            plt.figure()
            plt.scatter(points_at_z[:, 0], points_at_z[:, 1], s=1)
            plt.gca().set_aspect('equal', adjustable='box')

            # Remove axes
            plt.axis('off')
            image_path = os.path.join(folder_path, f'contour_{i}.png')
            plt.savefig(image_path)
            plt.close()
            saved_image_paths.append(image_path)

    return saved_image_paths

if __name__ == '__main__':
    folder_path = '/path/to/folder'  # Specify the folder path containing the .ply file
    num_contours = 50  # Specify the number of contour lines
    saved_image_paths = generate_topographic_map(folder_path, num_contours, num_iterations=4)