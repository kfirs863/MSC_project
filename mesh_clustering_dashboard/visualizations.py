import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def visualize_tsne_plotly(tsne_results, cluster_labels, subfolder_names):
    """Creates a t-SNE scatter plot with color by cluster label."""
    df = pd.DataFrame({
        't-SNE Dimension 1': tsne_results[:, 0],
        't-SNE Dimension 2': tsne_results[:, 1],
        'Cluster': cluster_labels,
        'Mesh': subfolder_names
    })
    fig = px.scatter(
        df,
        x='t-SNE Dimension 1',
        y='t-SNE Dimension 2',
        color='Cluster',
        title='t-SNE Clustering Results',
        hover_data=['Mesh']
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2'
    )
    return fig


def visualize_feature_importance_plotly(features, feature_names):
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    # Handle the trivial case
    if features.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough features for PCA-based importance!", showarrow=False)
        return fig

    pca = PCA(n_components=2)
    pca.fit(features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = go.Figure()
    for i, feature in enumerate(feature_names):
        x_val = float(loadings[i, 0])  # ensure standard Python float
        y_val = float(loadings[i, 1])  # ensure standard Python float
        feature_str = str(feature)      # ensure it's a standard Python str

        fig.add_trace(go.Scatter(
            x=[0, x_val],
            y=[0, y_val],
            mode='lines+markers+text',
            # Make sure name and text are plain strings:
            name=feature_str,
            text=[feature_str, ''],
            textposition='top center',
            showlegend=False,
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title='Feature Importance (PCA Loadings)',
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        showlegend=False,
        width=600,
        height=600,
        template='plotly_white'
    )
    return fig



def visualize_mesh3d(meshes, cluster_labels, selected_cluster=None, single_color='#FF0000'):
    """
    Visualize only the meshes belonging to selected_cluster with a single color.
    If selected_cluster is None, visualize all meshes in red.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Filter meshes if a cluster is specified
    if selected_cluster is not None:
        mesh_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == selected_cluster]
        filtered_meshes = [meshes[i] for i in mesh_indices]
    else:
        filtered_meshes = meshes

    all_x, all_y, all_z = [], [], []
    all_faces = []
    current_vertex_count = 0

    for mesh in filtered_meshes:
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        z = mesh.vertices[:, 2]

        faces = mesh.faces + current_vertex_count

        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
        all_faces.extend(faces)
        current_vertex_count += len(x)

    if len(all_x) == 0:
        # No meshes for this cluster
        return go.Figure()

    fig = go.Figure(data=[go.Mesh3d(
        x=all_x,
        y=all_y,
        z=all_z,
        i=np.array(all_faces)[:, 0],
        j=np.array(all_faces)[:, 1],
        k=np.array(all_faces)[:, 2],
        opacity=0.5,
        color=single_color,     # <--- use single_color
        showscale=False
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def visualize_surface_area_plotly(subfolder_names, surface_areas):
    df = pd.DataFrame({'Mesh': subfolder_names, 'Surface Area': surface_areas})
    fig = px.bar(df, x='Mesh', y='Surface Area',
                 title='Surface Areas of Meshes',
                 color='Surface Area', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='Surface Area')
    return fig


def visualize_sa_to_volume_ratio_plotly(ratios, subfolder_names):
    df = pd.DataFrame({'Mesh': subfolder_names, 'SA to Volume Ratio': ratios})
    fig = px.bar(df, x='Mesh', y='SA to Volume Ratio',
                 title='Surface Area to Volume Ratios',
                 color='SA to Volume Ratio', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='SA to Volume Ratio')
    return fig


def visualize_vertices_faces_plotly(num_vertices, num_faces, subfolder_names):
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Number of Vertices': num_vertices,
        'Number of Faces': num_faces
    })
    fig = px.scatter(df, x='Number of Vertices', y='Number of Faces',
                     text='Mesh', title='Mesh Complexity',
                     hover_data=['Mesh'])
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='Number of Vertices', yaxis_title='Number of Faces')
    return fig


def visualize_curvature_plotly(curvatures, title):
    """Box plot or distribution of curvature values."""
    if isinstance(curvatures, (list, np.ndarray)):
        flattened = []
        for val in curvatures:
            if isinstance(val, (list, np.ndarray)):
                flattened.extend(val)
            else:
                flattened.append(val)
    else:
        flattened = [curvatures]

    fig = go.Figure()
    fig.add_trace(go.Box(y=flattened, name=title))
    fig.update_layout(title=title)
    return fig


def visualize_edge_lengths_plotly(edge_lengths, title):
    """Histogram of edge lengths."""
    if isinstance(edge_lengths, (list, np.ndarray)):
        flattened = []
        for val in edge_lengths:
            if isinstance(val, (list, np.ndarray)):
                flattened.extend(val)
            else:
                flattened.append(val)
    else:
        flattened = [edge_lengths]

    df = pd.DataFrame({'Edge Length': flattened})
    fig = px.histogram(df, x='Edge Length', nbins=30,
                       title=f'Edge Length Distribution - {title}')
    return fig


def visualize_face_areas_plotly(face_areas, title):
    """Histogram of face areas."""
    flattened = []
    if isinstance(face_areas, (list, np.ndarray)):
        for val in face_areas:
            if isinstance(val, (list, np.ndarray)):
                flattened.extend(val)
            else:
                flattened.append(val)
    else:
        flattened.append(face_areas)

    df = pd.DataFrame({'Face Area': flattened})
    fig = px.histogram(df, x='Face Area', nbins=50,
                       title=f'Face Area Distribution - {title}')
    return fig


def visualize_axis_ratio_plotly(axis_ratios, subfolder_names):
    df = pd.DataFrame({
        'Mesh': subfolder_names,
        'Axis Ratio': axis_ratios
    })
    fig = px.bar(df, x='Mesh', y='Axis Ratio',
                 title='Axis Ratios of Artifacts',
                 color='Axis Ratio', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title='Mesh', yaxis_title='Axis Ratio (Major/Minor)')
    return fig


def visualize_cluster_in_grid(
    meshes,
    cluster_labels,
    subfolder_names,
    selected_cluster,
    num_columns=6,         # how many meshes per row
    spacing=1.0,           # horizontal/vertical spacing in XY
):
    """
    Visualize all meshes from `selected_cluster` side by side in a 3D grid.

    Steps per mesh:
    1) PCA => rotate smallest-variance axis to align with global Z-axis.
    2) Scale bounding-box largest dimension to 1.
    3) Shift bounding-box so z-min is at 0 (so we see full depth above XY-plane).
    4) Offset each mesh in X,Y to form a grid (row, col).
    5) Place a text label near the bounding-box center in 3D.
    """
    import plotly.graph_objects as go
    import numpy as np
    import trimesh
    from sklearn.decomposition import PCA

    # 1) Filter only meshes in the selected cluster
    mesh_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == selected_cluster]
    if not mesh_indices:
        return go.Figure()

    # Subset
    selected_meshes = [meshes[i] for i in mesh_indices]
    selected_names = [subfolder_names[i] for i in mesh_indices]

    # Arrays for a single Mesh3d trace
    all_x, all_y, all_z = [], [], []
    all_faces = []
    current_vertex_count = 0

    # Arrays for labeling (Scatter3d)
    label_x, label_y, label_z = [], [], []
    label_text = []

    for idx, (mesh, mesh_name) in enumerate(zip(selected_meshes, selected_names)):
        # Determine which row,col in the grid
        row = idx // num_columns
        col = idx % num_columns

        # Offsets in X,Y so each mesh sits in a unique grid cell
        x_offset = col * (1.0 + spacing)
        y_offset = -row * (1.0 + spacing)

        # Copy mesh so we don't mutate the original
        mesh_copy = mesh.copy()

        # --- (A) PCA to align smallest-variance axis => Z
        verts = mesh_copy.vertices
        pca = PCA(n_components=3)
        pca.fit(verts)
        # The smallest-variance axis is pca.components_[2]
        normal_axis = pca.components_[2]
        align_mat = trimesh.geometry.align_vectors(normal_axis, [0, 0, -1])
        mesh_copy.apply_transform(align_mat)

        # --- (B) Scale bounding box largest dimension => 1
        bb_min, bb_max = mesh_copy.bounds
        bb_size = bb_max - bb_min
        max_dim = bb_size.max()
        if max_dim > 0:
            scale_factor = 1.0 / max_dim
            mesh_copy.vertices *= scale_factor

        # --- (C) Shift so the bounding-box z-min is at 0
        # Recompute bounds
        bb_min2, bb_max2 = mesh_copy.bounds
        # We'll shift in *all* axes so bounding box min is (0,0,0).
        # That ensures the shape starts at z=0, so you can see the entire "depth" above the plane.
        mesh_copy.vertices -= bb_min2

        # --- (D) Now offset in X,Y for the grid. Keep Z as is.
        mesh_copy.vertices[:, 0] += x_offset
        mesh_copy.vertices[:, 1] += y_offset
        # We do NOT force Z=0; we keep the shape's real vertical extent.

        # Recompute bounding box for labeling
        new_bb_min, new_bb_max = mesh_copy.bounds
        center_x = 0.5 * (new_bb_min[0] + new_bb_max[0])
        center_y = 0.5 * (new_bb_min[1] + new_bb_max[1])
        center_z = 0.5 * (new_bb_min[2] + new_bb_max[2])  # mid-height

        # Store label info
        label_x.append(center_x)
        label_y.append(center_y)
        label_z.append(center_z)
        label_text.append(mesh_name)

        # Adjust face indices for single mesh3d
        faces = mesh_copy.faces + current_vertex_count

        # Append geometry
        all_x.extend(mesh_copy.vertices[:, 0])
        all_y.extend(mesh_copy.vertices[:, 1])
        all_z.extend(mesh_copy.vertices[:, 2])
        all_faces.extend(faces)
        current_vertex_count += len(mesh_copy.vertices)

    # (E) Build Mesh3d trace
    mesh_trace = go.Mesh3d(
        x=all_x,
        y=all_y,
        z=all_z,
        i=np.array(all_faces)[:, 0],
        j=np.array(all_faces)[:, 1],
        k=np.array(all_faces)[:, 2],
        color='lightblue',
        opacity=1.0,
        showscale=False,
        name="Meshes"
    )

    # (F) Build Scatter3d trace for text labels
    label_trace = go.Scatter3d(
        x=label_x,
        y=label_y,
        z=label_z,
        mode='text',
        text=label_text,
        textposition='top center',
        textfont=dict(color='black', size=14),
        name="Mesh Labels"
    )

    fig = go.Figure(data=[mesh_trace, label_trace])

    fig.update_layout(
        title=f"Cluster {selected_cluster} - 3D Grid (Depth Preserved)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=-2.5, z=2.5))  # tweak as desired
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig
