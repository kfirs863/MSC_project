# app.py

import json
import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import numpy as np
import time  # For timing the clustering process
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from data_processing import (
    load_meshes_and_images_from_folders,
    extract_geometric_features,
    extract_combined_contour_features,
    extract_image_features,
    normalize_mesh, initialize_resnet_model
)
from clustering import perform_clustering
from mesh_clustering_dashboard.utils import load_feature_descriptions
from visualizations import (
    visualize_surface_area_plotly,
    visualize_aspect_ratios_plotly,
    visualize_first_to_second_axis_ratios_plotly,
    visualize_sa_to_volume_ratio_plotly,
    visualize_vertices_faces_plotly,
    visualize_curvature_plotly,
    visualize_face_areas_plotly,
    visualize_edge_lengths_plotly,
    visualize_tsne_plotly,
    visualize_mesh3d,
    visualize_feature_importance_plotly  # Import the new function
)

# Load the ResNet50 model
model, transform = initialize_resnet_model()

# Load feature descriptions
feature_descriptions = load_feature_descriptions()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Mesh Clustering Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Mesh Clustering Dashboard"), width=12)
    ], justify='center', style={'marginTop': 20, 'marginBottom': 20}),

    dbc.Row([
        dbc.Col([
            html.H4("Dataset Folder Path"),
            dbc.Input(type="text", id='dataset_path', value='/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets', placeholder="Enter the dataset folder path"),
            html.Br(),
            html.H4("Geometric Feature Selection"),
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Geometric Features",
                    children=[
                        dbc.Checklist(
                            id='geometric_features',
                            options=[
                                {'label': 'Surface Area', 'value': 'surface_area'},
                                # Removed bbox_size and bbox_aspect_ratios as they are influenced by coordinates
                                {'label': 'SA to Bounding Box Volume Ratio', 'value': 'sa_to_bbox_volume_ratio'},
                                # Removed eigenvalues and eigenvalues_ratio
                                # Removed first_to_second_axis_ratio
                                {'label': 'Number of Vertices', 'value': 'num_vertices'},
                                {'label': 'Number of Faces', 'value': 'num_faces'},
                                {'label': 'Mean Curvature', 'value': 'mean_curvature'},
                                {'label': 'Std Curvature', 'value': 'std_curvature'},
                                {'label': 'Mean Edge Length', 'value': 'mean_edge_length'},
                                {'label': 'Std Edge Length', 'value': 'std_edge_length'},
                                {'label': 'Mean Face Area', 'value': 'mean_face_area'},
                                {'label': 'Std Face Area', 'value': 'std_face_area'},
                            ],
                            value=[
                                'surface_area',
                                'sa_to_bbox_volume_ratio',
                                'num_vertices',
                                'num_faces',
                                'mean_curvature',
                                'std_curvature',
                                'mean_edge_length',
                                'std_edge_length',
                                'mean_face_area',
                                'std_face_area',
                            ],
                            inline=True
                        ),
                        html.Div(id='feature_info', style={'marginTop': '10px'})
                    ]
                )
            ], start_collapsed=False),  # Start expanded
            html.Br(),
            html.H4("2D Feature Extraction"),
            dbc.Accordion([
                dbc.AccordionItem(
                    title="2D Features",
                    children=[
                        dbc.Checklist(
                            id='image_features',
                            options=[
                                {'label': 'Contour Features', 'value': 'contour_features'},
                                {'label': 'Depth Features', 'value': 'depth_features'},
                                {'label': 'Skeleton Features', 'value': 'skeleton_features'},
                            ],
                            value=[
                                'contour_features',
                                'depth_features',
                                'skeleton_features',
                            ],
                            inline=True
                        ),
                        html.Div("Choose features for extraction from images.", style={'marginTop': '10px'})
                    ]
                )
            ], start_collapsed=False),  # Start expanded
            html.Br(),
            html.H4("Clustering Algorithm"),
            dcc.Dropdown(
                id='clustering_algorithm',
                options=[
                    {'label': 'KMeans', 'value': 'kmeans'},
                    {'label': 'DBSCAN', 'value': 'dbscan'},
                    {'label': 'Agglomerative', 'value': 'agglomerative'},
                    {'label': 'Spectral', 'value': 'spectral'},
                    {'label': 'Gaussian Mixture Model', 'value': 'gmm'},
                ],
                value='kmeans',
                clearable=False
            ),
            html.Br(),
            html.Div(id='clustering_parameters'),
            html.Br(),
            dbc.Button("Run Clustering", id='run_button', color='primary'),
            html.Br(),
            html.Div(id='metrics_output'),
            html.Div(id='runtime_output', style={'marginTop': '20px'})  # To display runtime
        ], width=3),

        dbc.Col([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div([
                    dcc.Tabs(id='tabs', value='tab-1', children=[
                        dcc.Tab(label='2D Visualizations', children=[
                            html.Div(id='tabs-content')
                        ]),
                        dcc.Tab(label='Feature Importance', children=[
                            dcc.Graph(id='feature_importance_graph')
                        ]),
                        dcc.Tab(label='3D Visualization', children=[
                            dcc.Graph(
                                id='3d_mesh',
                                figure={},
                                style={'height': '80vh'}  # Increase height to 80% of viewport height
                            )
                        ]),
                    ]),
                ])
            )
        ], width=9)
    ]),
], fluid=True)

# Add the feature descriptions to the info section
@app.callback(
    Output('feature_info', 'children'),
    Input('geometric_features', 'value'),
)
def display_feature_info(selected_features):
    formatted_features = {
        'surface_area': 'Surface Area',
        'sa_to_bbox_volume_ratio': 'SA to Bounding Box Volume Ratio',
        'num_vertices': 'Number of Vertices',
        'num_faces': 'Number of Faces',
        'mean_curvature': 'Mean Curvature',
        'std_curvature': 'Std Curvature',
        'mean_edge_length': 'Mean Edge Length',
        'std_edge_length': 'Std Edge Length',
        'mean_face_area': 'Mean Face Area',
        'std_face_area': 'Std Face Area',
    }

    return [
        html.Div([
            html.B(formatted_features[feature] + ": "),
            html.Span(feature_descriptions.get(feature, 'No description available.'))
        ]) for feature in selected_features
    ]

# Callback to display clustering parameters based on selected algorithm
@app.callback(
    Output('clustering_parameters', 'children'),
    Input('clustering_algorithm', 'value')
)
def display_clustering_parameters(algorithm):
    if algorithm == 'kmeans':
        return dbc.Card([dbc.CardBody([
            dbc.Label("Number of Clusters"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'kmeans_n_clusters'}, value=5, min=1),
            dbc.Label("Number of Iterations"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'kmeans_max_iter'}, value=20, min=1),
            dbc.Label("Random State"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'kmeans_random_state'},  min=0),
        ])])
    elif algorithm == 'dbscan':
        return dbc.Card([dbc.CardBody([
            dbc.Label("Epsilon (eps)"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'dbscan_eps'}, value=0.5, step=0.1),
            dbc.Label("Minimum Samples"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'dbscan_min_samples'}, value=5, min=1),
        ])])
    elif algorithm == 'agglomerative':
        return dbc.Card([dbc.CardBody([
            dbc.Label("Number of Clusters"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'agglo_n_clusters'}, value=5, min=1),
            dbc.Label("Linkage"),
            dcc.Dropdown(
                id={'type': 'clustering_param', 'index': 'agglo_linkage'},
                options=[
                    {'label': 'Ward', 'value': 'ward'},
                    {'label': 'Complete', 'value': 'complete'},
                    {'label': 'Average', 'value': 'average'},
                    {'label': 'Single', 'value': 'single'},
                ],
                value='ward',
                clearable=False
            ),
        ])])
    elif algorithm == 'spectral':
        return dbc.Card([dbc.CardBody([
            dbc.Label("Number of Clusters"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'spectral_n_clusters'}, value=5, min=1),
            dbc.Label("Number of Components"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'spectral_n_components'}, value=100, min=1),
        ])])
    elif algorithm == 'gmm':
        return dbc.Card([dbc.CardBody([
            dbc.Label("Number of Components"),
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'gmm_n_components'}, value=5, min=1),
            dbc.Label("Covariance Type"),
            dcc.Dropdown(
                id={'type': 'clustering_param', 'index': 'gmm_covariance_type'},
                options=[
                    {'label': 'Full', 'value': 'full'},
                    {'label': 'Tied', 'value': 'tied'},
                    {'label': 'Diag', 'value': 'diag'},
                    {'label': 'Spherical', 'value': 'spherical'},
                ],
                value='full',
                clearable=False
            ),
        ])])
    else:
        return html.Div()

# Callback to run clustering when button is clicked
@app.callback(
    [
        Output('metrics_output', 'children'),
        Output('tabs-content', 'children'),
        Output('3d_mesh', 'figure'),
        Output('feature_importance_graph', 'figure'),  # New Output for feature importance
        Output('runtime_output', 'children')  # Output for runtime
    ],
    Input('run_button', 'n_clicks'),
    State('geometric_features', 'value'),
    State('image_features', 'value'),
    State('clustering_algorithm', 'value'),
    State({'type': 'clustering_param', 'index': ALL}, 'value'),
    State('dataset_path', 'value'),
    prevent_initial_call=True
)
def run_clustering(n_clicks, selected_geometric_features, selected_image_features, algorithm, clustering_param_values,
                   dataset_path):
    if n_clicks is None or not dataset_path:
        return "", dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Start timing
    start_time = time.time()

    # Initialize parameters dict
    params = {}

    # Extract relevant parameters based on the selected algorithm
    if algorithm == 'kmeans':
        try:
            kmeans_n_clusters = int(clustering_param_values[0])
            kmeans_max_iter = int(clustering_param_values[1])
            kmeans_random_state = int(clustering_param_values[2]) if clustering_param_values[2] else None
        except (IndexError, ValueError):
            kmeans_n_clusters = 5
            kmeans_max_iter = 20
            kmeans_random_state = None
        params = {
            'n_clusters': kmeans_n_clusters,
            'max_iter': kmeans_max_iter,
            'random_state': kmeans_random_state
        }
    elif algorithm == 'dbscan':
        try:
            dbscan_eps = float(clustering_param_values[0])
            dbscan_min_samples = int(clustering_param_values[1])
        except (IndexError, ValueError):
            dbscan_eps = 0.5
            dbscan_min_samples = 5
        params = {
            'eps': dbscan_eps,
            'min_samples': dbscan_min_samples
        }
    elif algorithm == 'agglomerative':
        try:
            agglo_n_clusters = int(clustering_param_values[0])
            agglo_linkage = clustering_param_values[1]
        except (IndexError, ValueError):
            agglo_n_clusters = 5
            agglo_linkage = 'ward'
        params = {
            'n_clusters': agglo_n_clusters,
            'linkage': agglo_linkage
        }
    elif algorithm == 'spectral':
        try:
            spectral_n_clusters = int(clustering_param_values[0])
            spectral_n_components = int(clustering_param_values[1])
        except (IndexError, ValueError):
            spectral_n_clusters = 5
            spectral_n_components = 100
        params = {
            'n_clusters': spectral_n_clusters,
            'n_components': spectral_n_components
        }
    elif algorithm == 'gmm':
        try:
            gmm_n_components = int(clustering_param_values[0])
            gmm_covariance_type = clustering_param_values[1]
        except (IndexError, ValueError):
            gmm_n_components = 5
            gmm_covariance_type = 'full'
        params = {
            'n_components': gmm_n_components,
            'covariance_type': gmm_covariance_type
        }

    # Load the meshes and corresponding images from the user-defined dataset path
    meshes, normalized_meshes, depth_images, contour_images_list, skeleton_images, subfolder_names = load_meshes_and_images_from_folders(
        dataset_path)

    # Extract geometric features
    geometric_features, feature_data, feature_names = extract_geometric_features(normalized_meshes)

    # Initialize features list for combining
    features_list = []

    # Extract selected 2D features
    if 'contour_features' in selected_image_features:
        contour_features = extract_combined_contour_features(contour_images_list, model, transform)
        features_list.append(contour_features)

    if 'depth_features' in selected_image_features:
        depth_features = extract_image_features(depth_images, model, transform)
        features_list.append(depth_features)

    if 'skeleton_features' in selected_image_features:
        skeleton_features = extract_image_features(skeleton_images, model, transform)
        features_list.append(skeleton_features)

    # Map feature names to their indices
    feature_indices = [feature_names.index(feature) for feature in selected_geometric_features if feature in feature_names]
    selected_feature_names = [feature_names[index] for index in feature_indices]

    # Select the geometric features
    selected_geometric_features_array = geometric_features[:, feature_indices] if feature_indices else np.empty(
        (geometric_features.shape[0], 0))

    # Combine the selected geometric features with the 2D features
    combined_features = np.hstack(
        features_list + [selected_geometric_features_array]) if features_list else selected_geometric_features_array

    # Standardize the combined features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Keep 95% variance
    reduced_features = pca.fit_transform(scaled_features)

    # Perform clustering
    cluster_labels, clusterer = perform_clustering(algorithm, params, reduced_features)

    # Calculate clustering metrics
    metrics = {}
    unique_clusters = set(cluster_labels)
    if len(unique_clusters) > 1 and (-1 not in unique_clusters or len(unique_clusters) > 2):
        silhouette_avg = silhouette_score(reduced_features, cluster_labels)
        db_score = davies_bouldin_score(reduced_features, cluster_labels)
        metrics['Silhouette Score'] = silhouette_avg
        metrics['Davies-Bouldin Index'] = db_score
    else:
        metrics['Silhouette Score'] = float('nan')
        metrics['Davies-Bouldin Index'] = float('nan')
    metrics['Clustering Algorithm'] = algorithm.capitalize()

    # Prepare 3D Mesh Visualization
    fig_mesh3d = visualize_mesh3d(meshes, cluster_labels)

    # --- t-SNE Visualization ---
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)

    # Prepare visualizations
    figures = []

    # 1. t-SNE Plot
    fig_tsne = visualize_tsne_plotly(tsne_results, cluster_labels, subfolder_names)
    fig_tsne.update_layout(
        title='t-SNE Clustering Results',
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2'
    )
    figures.append(fig_tsne)

    # 2. Surface Area
    if 'surface_area' in selected_geometric_features:
        surface_areas = [data['surface_area'] for data in feature_data]
        fig_surface_area = visualize_surface_area_plotly(subfolder_names, surface_areas)
        figures.append(fig_surface_area)

    # 3. SA to Volume Ratio
    if 'sa_to_bbox_volume_ratio' in selected_geometric_features:
        sa_to_volume_ratios = [data['sa_to_bbox_volume_ratio'] for data in feature_data]
        fig_sa_to_volume = visualize_sa_to_volume_ratio_plotly(sa_to_volume_ratios, subfolder_names)
        figures.append(fig_sa_to_volume)

    # 4. Number of Vertices and Faces
    if 'num_vertices' in selected_geometric_features or 'num_faces' in selected_geometric_features:
        num_vertices = [data['num_vertices'] for data in feature_data] if 'num_vertices' in selected_geometric_features else [0] * len(subfolder_names)
        num_faces = [data['num_faces'] for data in feature_data] if 'num_faces' in selected_geometric_features else [0] * len(subfolder_names)
        fig_vertices_faces = visualize_vertices_faces_plotly(num_vertices, num_faces, subfolder_names)
        figures.append(fig_vertices_faces)

    # 5. Curvature
    if 'mean_curvature' in selected_geometric_features or 'std_curvature' in selected_geometric_features:
        curvatures = [data['curvature'] for data in feature_data]  # Corrected line

        if 'mean_curvature' in selected_geometric_features:
            mean_curvatures = [np.mean(curv) for curv in curvatures]
            fig_mean_curvature = visualize_curvature_plotly(mean_curvatures, 'Mean Curvature')
            figures.append(fig_mean_curvature)

        if 'std_curvature' in selected_geometric_features:
            std_curvatures = [np.std(curv) for curv in curvatures]
            fig_std_curvature = visualize_curvature_plotly(std_curvatures, 'Std Curvature')
            figures.append(fig_std_curvature)

    # 6. Edge Lengths
    if 'mean_edge_length' in selected_geometric_features or 'std_edge_length' in selected_geometric_features:
        edge_lengths = [data['edge_lengths'] for data in feature_data]  # Corrected line
        if 'mean_edge_length' in selected_geometric_features:
            mean_edge_lengths = [np.mean(edges) for edges in edge_lengths]
            fig_mean_edge_length = visualize_edge_lengths_plotly(mean_edge_lengths, 'Mean Edge Length')
            figures.append(fig_mean_edge_length)
        if 'std_edge_length' in selected_geometric_features:
            std_edge_lengths = [np.std(edges) for edges in edge_lengths]
            fig_std_edge_length = visualize_edge_lengths_plotly(std_edge_lengths, 'Std Edge Length')
            figures.append(fig_std_edge_length)

    # 7. Face Areas
    if 'mean_face_area' in selected_geometric_features or 'std_face_area' in selected_geometric_features:
        face_areas = [data['face_areas'] for data in feature_data]  # Corrected line
        if 'mean_face_area' in selected_geometric_features:
            mean_face_areas = [np.mean(areas) for areas in face_areas]
            fig_mean_face_area = visualize_face_areas_plotly(mean_face_areas, 'Mean Face Area')
            figures.append(fig_mean_face_area)
        if 'std_face_area' in selected_geometric_features:
            std_face_areas = [np.std(areas) for areas in face_areas]
            fig_std_face_area = visualize_face_areas_plotly(std_face_areas, 'Std Face Area')
            figures.append(fig_std_face_area)

    # --- Feature Importance Visualization ---
    feature_importance_fig = visualize_feature_importance_plotly(scaled_features, selected_feature_names)

    # Compile metrics
    metrics_display = dbc.Card([
        dbc.CardHeader("Clustering Evaluation Metrics"),
        dbc.CardBody([
            html.Ul([
                html.Li(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")
                for key, value in metrics.items()
            ])
        ])
    ])

    # Prepare tabs content for 2D Visualizations
    tabs_content = []
    for i, fig in enumerate(figures):
        tab_label = fig.layout.title.text if fig.layout.title else f'Tab {i + 1}'
        tabs_content.append(dcc.Tab(label=tab_label, children=[dcc.Graph(figure=fig)]))

    # Stop timing
    end_time = time.time()
    run_time = end_time - start_time

    return metrics_display, dbc.Tabs(children=tabs_content), fig_mesh3d, feature_importance_fig, f"Runtime: {run_time:.2f} seconds"


if __name__ == '__main__':
    app.run_server(debug=True)
