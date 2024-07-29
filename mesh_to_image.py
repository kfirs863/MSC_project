from pathlib import Path

import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from open3d.cuda.pybind.geometry import TriangleMesh
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

def load_mesh(filename):
    """ Load a mesh from an OBJ file and return its vertices and normals. """
    mesh: TriangleMesh = o3d.io.read_triangle_mesh(filename)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)  # Assume normals are already normalized
    return vertices, normals

def load_textured_mesh(filename):
    """ Load a textured mesh from an OBJ file. """
    mesh = o3d.io.read_triangle_mesh(filename)
    if not mesh.has_vertex_colors() and not mesh.textures:
        raise ValueError("The mesh does not have any vertex colors or textures.")
    return mesh

def rotate_point_cloud(vertices, rotation_matrix):
    """ Apply a rotation to the point cloud. """
    return np.dot(vertices, rotation_matrix.T)

def compute_optimal_rotation(normals, primary_normal_index=0):
    # Aim to align the primary normal with the z-axis ([0, 0, 1])
    primary_normal = normals[primary_normal_index]
    axis_of_rotation = np.cross(primary_normal, [0, 0, 1])
    angle_of_rotation = np.arccos(np.clip(np.dot(primary_normal, [0, 0, 1]), -1.0, 1.0))
    rotation_vector = axis_of_rotation * angle_of_rotation
    rotation = R.from_rotvec(rotation_vector)
    return rotation.as_matrix()

def align_point_cloud(point_cloud, normals, primary_normal_index=0):
    rotation_matrix = compute_optimal_rotation(normals, primary_normal_index)
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return rotated_point_cloud

def visualize_point_cloud(vertices):
    """ Visualize the point cloud. """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries([pcd])

    # Save as an image without any axis, colorbar, or white edges, using o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1600)

    # disable the axis and dots
    vis.get_render_option().show_coordinate_frame = False

    vis.add_geometry(pcd)
    vis.poll_events()
    vis.capture_screen_image('point_cloud.png')
    vis.destroy_window()


def generate_ortho_image(point_cloud, colors, resolution=(1000, 1000)):
    """ Generate an orthoimage from a point cloud. """
    res_x, res_y = resolution

    min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

    ix = np.floor((point_cloud[:, 0] - min_x) / (max_x - min_x) * (res_x - 1)).astype(int)
    iy = np.floor((point_cloud[:, 1] - min_y) / (max_y - min_y) * (res_y - 1)).astype(int)

    image = np.zeros((res_y, res_x, 3), dtype=np.uint8)
    image[iy, ix] = colors

    return image, ix, iy, min_x, max_x, min_y, max_y


def generate_ortho_image_neg_z(point_cloud, colors, resolution=(1000, 1000)):
    """ Generate an orthoimage from a point cloud viewed from -z direction. """
    # Define the resolution of the image
    res_x, res_y = resolution

    # Find min and max coordinates for normalization
    min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

    # Normalize point coordinates to image coordinates
    ix = np.floor((point_cloud[:, 0] - min_x) / (max_x - min_x) * (res_x - 1)).astype(int)
    iy = res_y - 1 - np.floor((point_cloud[:, 1] - min_y) / (max_y - min_y) * (res_y - 1)).astype(int)  # Flipped

    # Create the orthoimage with RGB channels
    image = np.zeros((res_y, res_x, 3), dtype=np.uint8)

    # Create a buffer to store color accumulations and counts
    color_buffer = np.zeros_like(image, dtype=np.uint32)
    count_buffer = np.zeros((res_y, res_x), dtype=np.uint32)

    # Accumulate colors and counts
    for i, (x, y) in enumerate(zip(ix, iy)):
        color_buffer[y, x] += colors[i]
        count_buffer[y, x] += 1

    # Calculate the average color where count is non-zero
    non_zero_mask = count_buffer > 0
    image[non_zero_mask] = (color_buffer[non_zero_mask] // count_buffer[non_zero_mask].reshape(-1, 1))

    return image


def generate_spherical_image(point_cloud, normals, resolution_y=3000):
    # Ensure the point cloud is centered on the origin
    center_coordinates = np.mean(point_cloud, axis=0)
    point_cloud -= center_coordinates

    # Align normals if necessary (this step may be skipped if not needed)
    # Compute rotation to align a primary normal with the z-axis
    primary_normal_index = np.argmax(point_cloud[:, 2])  # For example, the highest point
    point_cloud = align_point_cloud(point_cloud, normals, primary_normal_index)

    # Convert to spherical coordinates
    r = np.linalg.norm(point_cloud, axis=1)
    theta = np.arctan2(point_cloud[:,1], point_cloud[:,0])
    phi = np.arccos(point_cloud[:,2] / r)

    # Map spherical coordinates to 2D image coordinates
    x = (2.0 * resolution_y * (theta + np.pi) / (2 * np.pi)).astype(np.int32)
    y = (resolution_y * phi / np.pi).astype(np.int32)

    # Create the spherical image
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    # Assign points to the image, considering the aspect ratio
    for idx, (ix, iy) in enumerate(zip(x, y)):
        if 0 <= ix < resolution_x and 0 <= iy < resolution_y:
            image[iy, ix] = [255, 255, 255]  # Or the color corresponding to this point

    return image


def find_rotation_matrix(vertices):
    """ Find the rotation matrix to align the dominant plane of the point cloud with the XY plane. """
    pca = PCA(n_components=3)
    pca.fit(vertices)
    normal = pca.components_[2]  # The normal to the plane is the last principal component
    z_axis = np.array([0, 0, 1])

    # Compute the rotation matrix to align the normal with the z-axis
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    s = np.linalg.norm(v)

    if s == 0:  # If the normal is already aligned with the z-axis
        rotation_matrix = np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def extract_vertex_colors_from_texture(mesh):
    """ Extract vertex colors from the mesh texture. """
    mesh.compute_vertex_normals()
    if mesh.has_vertex_colors():
        return np.asarray(mesh.vertex_colors)
    elif mesh.textures:
        # Extract the UV coordinates
        uvs = np.asarray(mesh.triangle_uvs)
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        texture = np.asarray(mesh.textures[1])

        vertex_colors = np.zeros((len(vertices), 3))

        height, width = texture.shape[:2]

        for i in range(len(triangles)):
            for j in range(3):
                vertex_idx = triangles[i, j]
                uv = uvs[vertex_idx]  # Correct index calculation
                u = min(max(int(uv[0] * width), 0), width - 1)
                v = min(max(int((1 - uv[1]) * height), 0),
                        height - 1)  # Note the (1 - uv[1]) to account for different texture coordinate origin
                vertex_colors[vertex_idx] = texture[v, u, :3] / 255.0  # Normalize colors to [0, 1]

        return vertex_colors
    else:
        raise ValueError("The mesh does not have textures or vertex colors.")

if __name__ == '__main__':
    import base64
    # Usage
    filename = Path('/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/source/cross2.obj')
    # filename = Path('/mobileye/RPT/users/kfirs/temp/S01/S01.obj')

    vertices, normals = load_mesh(str(filename))
    # mesh = load_textured_mesh(str(filename))

    # vertices = np.asarray(mesh.vertices)
    # vertex_colors = extract_vertex_colors_from_texture(mesh)

    # if len(vertex_colors) == 0:
    #     raise ValueError("No vertex colors could be extracted from the texture.")

    # Determine if rotation is needed
    rotation_matrix = find_rotation_matrix(vertices)
    rotated_vertices = rotate_point_cloud(vertices, rotation_matrix)
    # rotated_colors = rotate_point_cloud(vertex_colors, rotation_matrix)


    # visualize_point_cloud(rotated_vertices)
    # spherical_image = generate_spherical_image(vertices, normals)
    #
    # # Save the spherical image without any axis, colorbar, or white edges
    # plt.imshow(spherical_image)
    # plt.axis('off')  # Disable axis
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the plot edges
    # plt.savefig(f'./images/{filename.stem}_spherical_projection.png', bbox_inches='tight', pad_inches=0, dpi=400)
    # plt.close()  # Close the plot to free memory

    colors = np.full((len(rotated_vertices ), 3), [255, 255, 255], dtype=np.uint8)  # White colors for all vertices

    # Generate orthoimage
    resolution = (2048, 2048)  # Set the desired resolution for the orthoimage
    ortho_image, ix, iy, min_x, max_x, min_y, max_y = generate_ortho_image(rotated_vertices, colors, resolution)

    # Save the orthoimage without any axis, colorbar, or white edges
    plt.imshow(ortho_image)
    plt.axis('off')  # Disable axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Adjust the plot edges
    plt.savefig( f'./images/{filename.stem}_orthoimage.png', bbox_inches='tight', pad_inches=0, dpi=600)  # Save the image
    plt.close()  # Close the plot to free memory
