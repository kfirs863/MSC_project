import numpy as np
import open3d
import open3d as o3d
from matplotlib import pyplot as plt


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Parameters:
    - x, y, z: Cartesian coordinates.

    Returns:
    - r, theta, phi: Spherical coordinates.
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    Parameters:
    - r: Radius.
    - theta: Azimuthal angle.
    - phi: Polar angle.

    Returns:
    - x, y, z: Cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def project_point_cloud_to_sphere(point_cloud_data):
    """
    Project a 3D point cloud onto a unit sphere.

    Parameters:
    - point_cloud_data: numpy array of shape (n, 3) where n is the number of points and each point has x, y, z coordinates.

    Returns:
    - spherical_point_cloud: numpy array of shape (n, 3) representing points on the sphere.
    """
    spherical_point_cloud = np.zeros_like(point_cloud_data)
    for i, (x, y, z) in enumerate(point_cloud_data):
        r, theta, phi = cartesian_to_spherical(x, y, z)
        x_sph, y_sph, z_sph = spherical_to_cartesian(1, theta, phi)
        spherical_point_cloud[i] = [x_sph, y_sph, z_sph]
    return spherical_point_cloud


def plot_point_cloud(point_cloud_data, title='3D Point Cloud'):
    """
    Plot a 3D point cloud using matplotlib.

    Parameters:
    - point_cloud_data: numpy array of shape (n, 3).
    - title: title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], point_cloud_data[:, 2], s=1)
    ax.set_title(title)
    plt.show()


def generate_spherical_image(center_coordinates, point_cloud, resolution_y=500):
    # Translate the point cloud by the negation of the center coordinates
    translated_points = point_cloud - center_coordinates

    # Convert 3D point cloud to spherical coordinates
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
    phi = np.arccos(translated_points[:, 2] / np.linalg.norm(translated_points, axis=1))

    # Map spherical coordinates to pixel coordinates
    x = (theta + np.pi) / (2 * np.pi) * (2 * resolution_y)
    y = phi / np.pi * resolution_y
    z = np.linalg.norm(translated_points, axis=1)

    # Create the spherical image with grayscale colors
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 1), dtype=np.uint8)

    # Create the mapping between point cloud and image coordinates
    mapping = np.full((resolution_y, resolution_x), -1, dtype=int)

    # Assign points to the image pixels
    for i in range(len(translated_points)):
        ix = np.clip(int(x[i]), 0, resolution_x - 1)
        iy = np.clip(int(y[i]), 0, resolution_y - 1)
        if mapping[iy, ix] == -1 or np.linalg.norm(translated_points[i]) < np.linalg.norm(
                translated_points[mapping[iy, ix]]):
            mapping[iy, ix] = i
            image[iy, ix] = int(z[i] * 255)
    return image


if __name__ == '__main__':
    # Reading the mesh using Open3D
    mesh: open3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
        '/mobileye/RPT/users/kfirs/temp/left-collumn-church-of-the-holy-sepulchre/source/Pillar1/Pillar1.obj')


    # Extract the point cloud from the mesh
    pcd: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(number_of_points=50000,use_triangle_normal=True)

    # Transforming the point cloud to Numpy
    pcd_np = np.asarray(pcd.points)

    # Project the point cloud onto a unit sphere
    spherical_image = generate_spherical_image(pcd_np, pcd.get_center())

    # Plotting with matplotlib
    fig = plt.figure(figsize=(np.shape(spherical_image)[1] / 72, np.shape(spherical_image)[0] / 72))
    fig.add_axes([0, 0, 1, 1])
    plt.imshow(spherical_image)
    plt.axis('off')

    # Saving to the disk
    plt.savefig("spherical_projection.jpg")

    # Plot the original point cloud
    plot_point_cloud(pcd_np, title='Original 3D Point Cloud')

    # Plot the spherical point cloud
    plot_point_cloud(spherical_pcd_np, title='Point Cloud Projected onto Sphere')
