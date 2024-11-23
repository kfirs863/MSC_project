import plotly.express as px
import pandas as pd
import numpy as np
import open3d as o3d
import trimesh
from matplotlib import pyplot as plt


def trimesh_to_open3d(mesh):
    """Convert Trimesh mesh to Open3D mesh."""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def visualize_surface_area(meshes, surface_areas, subfolder_names):
    """Visualizes surface areas of meshes using an interactive bar chart."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Surface Area': surface_areas
    })
    fig = px.bar(df, x='Mesh', y='Surface Area', title='Surface Areas of Meshes',
                 color='Surface Area', color_continuous_scale='Viridis')
    fig.show()


def visualize_aspect_ratios(aspect_ratios, subfolder_names):
    """Visualizes aspect ratios of meshes using an interactive grouped bar chart."""
    df = pd.DataFrame(aspect_ratios, columns=['Width/Height', 'Height/Depth', 'Width/Depth'])
    df['Mesh'] = subfolder_names
    fig = px.bar(df, x='Mesh', y=['Width/Height', 'Height/Depth', 'Width/Depth'],
                 title='Aspect Ratios of Artifacts', barmode='group')
    fig.show()


def visualize_sa_to_volume_ratio(ratios, subfolder_names):
    """Visualizes surface area to volume ratios using an interactive bar chart."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'SA to Volume Ratio': ratios
    })
    fig = px.bar(df, x='Mesh', y='SA to Volume Ratio', title='Surface Area to Volume Ratios of Artifacts',
                 color='SA to Volume Ratio', color_continuous_scale='Blues')
    fig.show()


def visualize_vertices_faces(num_vertices, num_faces, subfolder_names):
    """Visualizes the number of vertices and faces of meshes using an interactive scatter plot."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Number of Vertices': num_vertices,
        'Number of Faces': num_faces
    })
    fig = px.scatter(df, x='Number of Vertices', y='Number of Faces', text='Mesh',
                     title='Mesh Complexity of Artifacts')
    fig.update_traces(textposition='top center')
    fig.show()


def visualize_curvature(mesh, curvature):
    """Visualizes the distribution of curvature values using an interactive histogram."""
    df = pd.DataFrame({'Curvature': curvature})
    fig = px.histogram(df, x='Curvature', nbins=50, title='Curvature Distribution')
    fig.show()


def visualize_face_areas(mesh):
    """Visualize mesh with vertex colors based on face areas using Open3D."""
    # Convert Trimesh to Open3D mesh
    mesh_o3d = trimesh_to_open3d(mesh)

    # Compute face areas
    face_areas = mesh.area_faces
    face_areas_normalized = (face_areas - np.min(face_areas)) / (np.max(face_areas) - np.min(face_areas))

    # Open3D only supports vertex colors, so we'll assign colors to vertices based on face areas
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    # Create a vertex color array based on the average face area for each vertex
    vertex_colors = np.zeros((vertices.shape[0], 3))  # Initialize vertex color array with RGB values

    # Normalize colors based on face areas and assign to vertices
    for i, face in enumerate(faces):
        face_color = plt.cm.viridis(face_areas_normalized[i])[:3]  # Get color from colormap in RGB format
        vertex_colors[face] = face_color  # Assign the color to each vertex in the face

    # Set the vertex colors in the Open3D mesh
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Visualize the mesh with vertex coloring
    o3d.visualization.draw_geometries([mesh_o3d], window_name="Face Areas Visualization", mesh_show_back_face=True)


def visualize_edge_lengths(edge_lengths, subfolder_name):
    """Visualizes the distribution of edge lengths using an interactive histogram."""
    df = pd.DataFrame({'Edge Lengths': edge_lengths})
    fig = px.histogram(df, x='Edge Lengths', nbins=30, title=f'Edge Length Distribution for {subfolder_name}')
    fig.show()


def visualize_clustered_meshes(meshes, cluster_labels):
    """Visualize the clustered meshes in 3D, colored by their cluster labels using Open3D and save them."""
    # Create a unique color for each cluster
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.get_cmap('viridis', num_clusters)

    colored_meshes = []

    for i, mesh in enumerate(meshes):
        # Get RGB color from the colormap based on the cluster label
        color = colors(cluster_labels[i])[:3]  # Get RGB values in [0, 1] range
        color = np.array(color)  # Ensure it is an array

        # If the mesh is not already in Open3D format, convert it
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices),
                                                 triangles=o3d.utility.Vector3iVector(mesh.faces))
        else:
            mesh_o3d = mesh

        mesh_o3d.compute_vertex_normals()

        # Apply the color to the mesh
        mesh_o3d.paint_uniform_color(color)

        # Append the colored Open3D mesh to the list
        colored_meshes.append(mesh_o3d)

    # Visualize the meshes using Open3D
    o3d.visualization.draw_geometries(colored_meshes, window_name="Clustered Meshes", mesh_show_back_face=True)


def visualize_first_to_second_axis_ratios(feature_data, subfolder_names):
    """Visualizes the ratio of the first to second principal axes for each mesh interactively using Plotly."""
    # Extract the first-to-second axis ratio from feature_data
    ratios = [data['first_to_second_axis_ratio'] for data in feature_data]

    # Create a DataFrame for better handling in Plotly
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'First to Second Axis Ratio': ratios
    })

    # Create an interactive bar chart using Plotly
    fig = px.bar(df, x='Mesh', y='First to Second Axis Ratio',
                 title='Comparison of First to Second Principal Axis Ratios Across Meshes',
                 labels={'Mesh': 'Mesh', 'First to Second Axis Ratio': 'First/Second Axis Ratio'})

    # Update layout for better visibility
    fig.update_layout(xaxis_title='Mesh', yaxis_title='First to Second Principal Axis Ratio',
                      xaxis_tickangle=-45, height=600, width=900)

    # Display the interactive chart
    fig.show()