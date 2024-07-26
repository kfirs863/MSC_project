import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import Button
from pathlib import Path
import matplotlib.pyplot as plt

# Global variable to store transformation matrix
transformation_matrix = np.eye(4)


def load_mesh(filename):
    """Load a mesh from an OBJ file and return its vertices, normals, and colors."""
    mesh = o3d.io.read_triangle_mesh(filename)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors) * 255
    else:
        colors = np.ones_like(vertices) * 255  # default to white if no colors
    return vertices, normals, colors.astype(np.uint8)


def apply_transformation(point_cloud, transformation_matrix):
    """Apply a transformation matrix to the point cloud."""
    return np.dot(point_cloud, transformation_matrix[:3, :3].T) + transformation_matrix[:3, 3]


def cylindrical_projection(point_cloud, colors, resolution=(1000, 2000)):
    """Generate a 2D cylindrical projection image from a 3D point cloud."""
    # Ensure the point cloud is centered on the origin
    center_coordinates = np.mean(point_cloud, axis=0)
    point_cloud -= center_coordinates

    # Convert to cylindrical coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    z = z

    # Normalize cylindrical coordinates to image coordinates
    res_y, res_x = resolution
    ix = ((theta + np.pi) / (2 * np.pi) * (res_x - 1)).astype(int)
    iz = ((z - np.min(z)) / (np.max(z) - np.min(z)) * (res_y - 1)).astype(int)

    # Create the image
    image = np.zeros((res_y, res_x, 3), dtype=np.uint8)

    # Create a buffer to store color accumulations and counts
    color_buffer = np.zeros((res_y, res_x, 3), dtype=np.uint32)
    count_buffer = np.zeros((res_y, res_x), dtype=np.uint32)

    # Accumulate colors and counts
    for i in range(len(ix)):
        if 0 <= ix[i] < res_x and 0 <= iz[i] < res_y:
            color_buffer[iz[i], ix[i]] += colors[i]
            count_buffer[iz[i], ix[i]] += 1

    # Calculate the average color where count is non-zero
    non_zero_mask = count_buffer > 0
    image[non_zero_mask] = (color_buffer[non_zero_mask] // count_buffer[non_zero_mask].reshape(-1, 1))

    return image


def visualize_and_capture_transform(vertices):
    """Visualize the point cloud and capture the transformation matrix."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    global transformation_matrix

    def capture_transform(vis):
        nonlocal transformation_matrix
        transformation_matrix = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
        vis.close()

    vis.register_key_callback(ord("C"), capture_transform)
    vis.run()
    vis.destroy_window()

    return transformation_matrix


def on_button_click():
    """Callback for the button to capture the transformation matrix."""
    global transformation_matrix
    transformation_matrix = visualize_and_capture_transform(vertices)
    root.quit()


if __name__ == '__main__':
    # Load the mesh
    filename = Path(
        '/mobileye/RPT/users/kfirs/temp/left-collumn-church-of-the-holy-sepulchre/source/Pillar1/Pillar1.obj')
    vertices, normals, colors = load_mesh(str(filename))

    # Create Tkinter GUI
    root = tk.Tk()
    root.geometry("300x100")
    root.title("3D Object Viewer")

    # Display button to capture the transformation
    button = Button(root, text="Align Object and Press 'C' to Capture", command=on_button_click)
    button.pack(pady=20)

    root.mainloop()

    # Apply the transformation to the point cloud
    transformed_vertices = apply_transformation(vertices, transformation_matrix)

    # Generate the cylindrical projection image
    cylindrical_image = cylindrical_projection(transformed_vertices, colors, resolution=(2000, 4000))

    # Display the image
    plt.imshow(cylindrical_image)
    plt.axis('off')  # Hide axes
    plt.show()
