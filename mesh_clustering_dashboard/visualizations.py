# visualizations.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def visualize_tsne_plotly(tsne_results, cluster_labels, subfolder_names):
    """Creates a square t-SNE scatter plot."""
    df = pd.DataFrame({
        't-SNE Dimension 1': tsne_results[:, 0],
        't-SNE Dimension 2': tsne_results[:, 1],
        'Cluster': cluster_labels,
        'Mesh': subfolder_names
    })
    fig = px.scatter(df, x='t-SNE Dimension 1', y='t-SNE Dimension 2', color='Cluster',
                     title='t-SNE Clustering Results', labels={'Cluster': 'Cluster Label'},
                     hover_data=['Mesh'])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2')

    # Set aspect ratio to be equal
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )

    return fig


def visualize_surface_area_plotly(subfolder_names, surface_areas):
    """Creates a Plotly bar chart for surface areas."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Surface Area': surface_areas
    })
    fig = px.bar(df, x='Mesh', y='Surface Area', title='Surface Areas of Meshes',
                color='Surface Area', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='Surface Area')
    return fig

def visualize_aspect_ratios_plotly(aspect_ratios, subfolder_names):
    """Creates a Plotly grouped bar chart for aspect ratios."""
    df = pd.DataFrame(aspect_ratios, columns=['Width/Height', 'Height/Depth', 'Width/Depth'])
    df['Mesh'] = subfolder_names
    fig = px.bar(df, x='Mesh', y=['Width/Height', 'Height/Depth', 'Width/Depth'],
                 title='Aspect Ratios of Artifacts', barmode='group',
                 labels={'value': 'Aspect Ratio', 'variable': 'Ratio Type'})
    fig.update_layout(xaxis_title='Mesh', yaxis_title='Aspect Ratio')
    return fig

def visualize_first_to_second_axis_ratios_plotly(ratios, subfolder_names):
    """Creates a Plotly bar chart for first to second principal axis ratios."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'First to Second Axis Ratio': ratios
    })
    fig = px.bar(df, x='Mesh', y='First to Second Axis Ratio',
                 title='First to Second Principal Axis Ratios',
                 labels={'Mesh': 'Mesh', 'First to Second Axis Ratio': 'First/Second Axis Ratio'},
                 color='First to Second Axis Ratio', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='First to Second Axis Ratio')
    return fig

def visualize_sa_to_volume_ratio_plotly(ratios, subfolder_names):
    """Creates a Plotly bar chart for surface area to volume ratios."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'SA to Volume Ratio': ratios
    })
    fig = px.bar(df, x='Mesh', y='SA to Volume Ratio',
                 title='Surface Area to Volume Ratios of Artifacts',
                 color='SA to Volume Ratio', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='SA to Volume Ratio')
    return fig

def visualize_vertices_faces_plotly(num_vertices, num_faces, subfolder_names):
    """Creates a Plotly scatter plot for number of vertices and faces."""
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Number of Vertices': num_vertices,
        'Number of Faces': num_faces
    })
    fig = px.scatter(df, x='Number of Vertices', y='Number of Faces', text='Mesh',
                     title='Mesh Complexity of Artifacts',
                     hover_data=['Mesh'])
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='Number of Vertices', yaxis_title='Number of Faces')
    return fig


def visualize_curvature_plotly(curvatures, title):
    if isinstance(curvatures, (list, np.ndarray)):
        flat_curvatures = [curv for curv in curvatures if isinstance(curv, (list, np.ndarray))]
        flat_curvatures = [item for sublist in flat_curvatures for item in sublist]  # Flatten
    else:
        flat_curvatures = [curvatures]  # Handle single value

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Box(y=flat_curvatures, name=title))
    fig.update_layout(title=title)

    return fig



def visualize_edge_lengths_plotly(edge_lengths, title):
    """Creates a Plotly histogram for edge length distribution."""
    flat_edge_lengths = []

    # Check if edge_lengths is a list of arrays or a single array
    if isinstance(edge_lengths, (list, np.ndarray)):
        for mesh_lengths in edge_lengths:
            if isinstance(mesh_lengths, (list, np.ndarray)):
                flat_edge_lengths.extend(mesh_lengths)
            else:
                flat_edge_lengths.append(mesh_lengths)  # Handle single values

    # Convert to DataFrame
    df = pd.DataFrame({'Edge Length': flat_edge_lengths})
    fig = px.histogram(df, x='Edge Length', nbins=30, title=f'Edge Length Distribution - {title}',
                       labels={'Edge Length': 'Edge Length'})
    return fig




def visualize_face_areas_plotly(face_areas, title):
    """Creates a Plotly histogram for face area distribution."""
    flat_face_areas = []

    # Ensure face_areas is treated as an iterable
    if isinstance(face_areas, (list, np.ndarray)):
        for mesh_areas in face_areas:
            if isinstance(mesh_areas, (list, np.ndarray)):
                flat_face_areas.extend(mesh_areas)  # Flatten if it's iterable
            else:
                flat_face_areas.append(mesh_areas)  # If it's a single value, add it
    else:
        flat_face_areas.append(face_areas)  # Handle single value case

    df = pd.DataFrame({'Face Area': flat_face_areas})
    fig = px.histogram(df, x='Face Area', nbins=50, title=f'Face Area Distribution - {title}',
                       labels={'Face Area': 'Face Area'})
    return fig





def visualize_mesh3d(meshes, cluster_labels):
    """Visualize multiple 3D meshes using Plotly's Mesh3d."""
    # Initialize lists for vertices, faces, and colors
    all_x, all_y, all_z = [], [], []
    all_faces = []
    color_values = []

    # Create a color map for cluster labels
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)

    # Generate a colorscale
    colorscale = 'Viridis'  # Choose a colorscale
    color_map = {label: idx for idx, label in enumerate(unique_labels)}

    current_vertex_count = 0  # To keep track of the vertex indices for faces

    for mesh, label in zip(meshes, cluster_labels):
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        z = mesh.vertices[:, 2]

        # Append vertices
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)

        # Append faces, adjusting indices to account for previously added vertices
        faces = mesh.faces + current_vertex_count
        all_faces.extend(faces)

        # Map the color for this mesh based on its cluster label
        color_values.extend([color_map[label]] * len(x))

        # Update the vertex count
        current_vertex_count += len(x)

    # Create the figure
    fig = go.Figure(data=[go.Mesh3d(
        x=all_x,
        y=all_y,
        z=all_z,
        i=np.array(all_faces)[:, 0],
        j=np.array(all_faces)[:, 1],
        k=np.array(all_faces)[:, 2],
        opacity=0.5,
        colorscale=colorscale,
        intensity=color_values,  # Use the mapped color values for intensity
        showscale=True  # Show color scale
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ))

    return fig


