import json
import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import numpy as np
import time
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from data_processing import (
    load_meshes_from_folders,
    extract_geometric_features,
    load_feature_descriptions
)
from clustering import perform_clustering
from visualizations import (
    # We'll rely on your existing functions, including:
    visualize_cluster_in_grid,   # Important: uses original Z-values
)

# Global placeholders (populated after clustering)
meshes = []
cluster_labels = []
subfolder_names = []
feature_data = []

# Load feature descriptions (for the feature info panel)
feature_descriptions = load_feature_descriptions()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Mesh Clustering Dashboard"

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Mesh Clustering Dashboard"), width=12)
    ], justify='center', style={'marginTop': 20, 'marginBottom': 20}),

    # Hidden store to keep cluster data after clustering
    dcc.Store(id='cluster_data_store'),

    dbc.Row([
        # --- Left Column (Controls) ---
        dbc.Col([
            html.H4("Dataset Folder Path"),
            dbc.Input(
                type="text",
                id='dataset_path',
                value='/path/to/your/datasets',  # put your default path here
                placeholder="Enter the dataset folder path"
            ),
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
                                {'label': 'SA to Bounding Box Volume Ratio', 'value': 'sa_to_bbox_volume_ratio'},
                                {'label': 'Number of Vertices', 'value': 'num_vertices'},
                                {'label': 'Number of Faces', 'value': 'num_faces'},
                                {'label': 'Mean Curvature', 'value': 'mean_curvature'},
                                {'label': 'Std Curvature', 'value': 'std_curvature'},
                                {'label': 'Mean Edge Length', 'value': 'mean_edge_length'},
                                {'label': 'Std Edge Length', 'value': 'std_edge_length'},
                                {'label': 'Mean Face Area', 'value': 'mean_face_area'},
                                {'label': 'Std Face Area', 'value': 'std_face_area'},
                                {'label': 'Axis Ratio', 'value': 'axis_ratio'},
                                {'label': 'Median Depth', 'value': 'median_depth'},
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
                                'axis_ratio',
                                'median_depth',
                            ],
                            inline=True
                        ),
                        html.Div(id='feature_info', style={'marginTop': '10px'})
                    ]
                )
            ], start_collapsed=False),
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
            html.Div(id='runtime_output', style={'marginTop': '20px'})
        ], width=2),  # <--- Make this narrower so we have more space for the 3D area

        # --- Right Column: Tabs / 3D Visuals ---
        dbc.Col([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div([
                    dcc.Tabs(id='tabs', value='tab-1', children=[

                        # 1) t-SNE Visualization Tab
                        dcc.Tab(label='t-SNE Visualization', children=[
                            html.Div([
                                dcc.Graph(
                                    id='tsne-graph',
                                    figure={},
                                    style={'height': '70vh'}  # increase height
                                ),
                                html.Div([
                                    html.H4("What is t-SNE?"),
                                    dcc.Markdown("""
**t-SNE** (t-distributed Stochastic Neighbor Embedding) is a technique for 
reducing high-dimensional data (multiple geometric features) down to **2D**, 
while trying to preserve how similar/dissimilar data points are 
in the original feature space.

- Points close together in the 2D plot => similar in chosen features
- Points far apart => dissimilar

It helps visualize if your chosen features + clustering produce well-separated groups.
                                    """, style={'margin-top': '20px'})
                                ], style={'margin-top': '20px'})
                            ])
                        ]),

                        # 2) Side-by-side 3D Visualization (Grids)
                        dcc.Tab(label='3D Visualization (Grids)', children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H4('Left View (Cluster)'),
                                    dcc.Dropdown(
                                        id='left-cluster-dropdown',
                                        placeholder='Select cluster (left)'
                                    ),
                                    dcc.Graph(
                                        id='left-3d-mesh',
                                        figure={},
                                        style={'height': '80vh'}  # bigger 3D area
                                    )
                                ], width=6),

                                dbc.Col([
                                    html.H4('Right View (Cluster)'),
                                    dcc.Dropdown(
                                        id='right-cluster-dropdown',
                                        placeholder='Select cluster (right)'
                                    ),
                                    dcc.Graph(
                                        id='right-3d-mesh',
                                        figure={},
                                        style={'height': '80vh'}  # bigger 3D area
                                    )
                                ], width=6),
                            ])
                        ]),

                    ]),
                ])
            )
        ], width=10)  # <--- Give more width to the 3D visuals
    ]),
], fluid=True)


# --- Callbacks ---

# 1) Display feature info
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
        'axis_ratio': 'Axis Ratio',
        'median_depth': 'Median Depth'
    }

    return [
        html.Div([
            html.B(f"{formatted_features[feature]}: "),
            html.Span(feature_descriptions.get(feature, 'No description available.'))
        ]) for feature in selected_features
    ]

# 2) Show parameters for chosen algorithm
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
            dbc.Input(type="number", id={'type': 'clustering_param', 'index': 'kmeans_random_state'}, min=0),
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

# 3) Main "Run Clustering" Callback
@app.callback(
    [
        Output('metrics_output', 'children'),
        Output('tsne-graph', 'figure'),
        Output('runtime_output', 'children'),
        Output('cluster_data_store', 'data')
    ],
    Input('run_button', 'n_clicks'),
    State('geometric_features', 'value'),
    State('clustering_algorithm', 'value'),
    State({'type': 'clustering_param', 'index': ALL}, 'value'),
    State('dataset_path', 'value'),
    prevent_initial_call=True
)
def run_clustering(n_clicks, selected_features, algorithm, param_values, dataset_path):
    global meshes, cluster_labels, subfolder_names, feature_data

    if not n_clicks or not dataset_path:
        return "", {}, "", {}

    start_time = time.time()

    # Build clustering params
    params = {}
    if algorithm == 'kmeans':
        try:
            k = int(param_values[0])
            max_iter = int(param_values[1])
            rand_state = int(param_values[2]) if param_values[2] else None
        except:
            k, max_iter, rand_state = 5, 20, None
        params = {
            'n_clusters': k,
            'max_iter': max_iter,
            'random_state': rand_state
        }
    elif algorithm == 'dbscan':
        try:
            eps = float(param_values[0])
            min_samp = int(param_values[1])
        except:
            eps, min_samp = 0.5, 5
        params = {
            'eps': eps,
            'min_samples': min_samp
        }
    elif algorithm == 'agglomerative':
        try:
            agg_k = int(param_values[0])
            agg_link = param_values[1]
        except:
            agg_k, agg_link = 5, 'ward'
        params = {
            'n_clusters': agg_k,
            'linkage': agg_link
        }
    elif algorithm == 'spectral':
        try:
            sp_k = int(param_values[0])
            sp_comp = int(param_values[1])
        except:
            sp_k, sp_comp = 5, 100
        params = {
            'n_clusters': sp_k,
            'n_components': sp_comp
        }
    elif algorithm == 'gmm':
        try:
            gmm_k = int(param_values[0])
            gmm_cov = param_values[1]
        except:
            gmm_k, gmm_cov = 5, 'full'
        params = {
            'n_components': gmm_k,
            'covariance_type': gmm_cov
        }

    # 1) Load
    meshes, subfolder_names = load_meshes_from_folders(dataset_path)

    # 2) Extract features
    geom_feats, feature_data, feat_names = extract_geometric_features(meshes)

    # 3) Filter user selection
    indices = [feat_names.index(f) for f in selected_features if f in feat_names]
    selected_arr = geom_feats[:, indices] if indices else np.empty((geom_feats.shape[0], 0))

    # 4) Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected_arr)

    # 5) Clustering
    cluster_labels, _ = perform_clustering(algorithm, params, scaled)

    # Build color map
    uniq_labels = sorted(set(cluster_labels))
    palette = px.colors.qualitative.Safe
    label2color = {}
    for i, lbl in enumerate(uniq_labels):
        color_idx = i % len(palette)
        label2color[int(lbl)] = palette[color_idx]

    # 6) Metrics
    metrics = {}
    if len(uniq_labels) > 1 and (-1 not in uniq_labels or len(uniq_labels) > 2):
        sil = silhouette_score(scaled, cluster_labels)
        dbs = davies_bouldin_score(scaled, cluster_labels)
        metrics['Silhouette Score'] = sil
        metrics['Davies-Bouldin Index'] = dbs
    else:
        metrics['Silhouette Score'] = float('nan')
        metrics['Davies-Bouldin Index'] = float('nan')
    metrics['Clustering Algorithm'] = algorithm.capitalize()

    # 7) t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled)
    df_labels = [str(lbl) for lbl in cluster_labels]
    color_map_str_keys = {str(k): v for k, v in label2color.items()}

    df_tsne = pd.DataFrame({
        't-SNE Dim1': tsne_results[:, 0],
        't-SNE Dim2': tsne_results[:, 1],
        'Cluster': df_labels,
        'Mesh': subfolder_names
    })

    fig_tsne = px.scatter(
        df_tsne,
        x='t-SNE Dim1',
        y='t-SNE Dim2',
        color='Cluster',
        hover_data=['Mesh'],
        color_discrete_map=color_map_str_keys
    )
    fig_tsne.update_layout(
        title='t-SNE Clustering Results',
        xaxis_title='t-SNE Dim1',
        yaxis_title='t-SNE Dim2'
    )

    # Metrics display
    metrics_card = dbc.Card([
        dbc.CardHeader("Clustering Evaluation Metrics"),
        dbc.CardBody([
            html.Ul([
                html.Li(f"{key}: {value:.4f}" if isinstance(value, (int,float)) else f"{key}: {value}")
                for key,value in metrics.items()
            ])
        ])
    ])

    end_time = time.time()
    run_time = end_time - start_time

    # cluster_data -> JSON-serializable
    cluster_data = {
        'cluster_labels': [int(lbl) for lbl in cluster_labels],
        'subfolder_names': [str(name) for name in subfolder_names],
        'label_to_color': {str(k):v for k,v in label2color.items()}
    }

    return metrics_card, fig_tsne, f"Runtime: {run_time:.2f} seconds", cluster_data


# 4) Populate cluster dropdowns
@app.callback(
    [Output('left-cluster-dropdown', 'options'),
     Output('right-cluster-dropdown', 'options')],
    Input('cluster_data_store', 'data')
)
def update_cluster_dropdowns(cluster_data):
    if not cluster_data:
        raise dash.exceptions.PreventUpdate

    labels = cluster_data['cluster_labels']
    unique_labels = sorted(set(labels))
    opts = [{'label':f"Cluster {lbl}", 'value':lbl} for lbl in unique_labels]
    return opts, opts


# 5) Side-by-side 3D visualization (using the 3D grid approach)
@app.callback(
    [Output('left-3d-mesh', 'figure'),
     Output('right-3d-mesh', 'figure')],
    [Input('left-cluster-dropdown', 'value'),
     Input('right-cluster-dropdown', 'value')],
    [State('cluster_data_store', 'data')]
)
def update_3d_grids(left_cluster, right_cluster, cluster_data):
    if not cluster_data or left_cluster is None or right_cluster is None:
        return {}, {}

    global meshes, cluster_labels, subfolder_names

    left_int = int(left_cluster)
    right_int = int(right_cluster)

    # Use the function from your "visualizations.py" that preserves Z
    left_fig = visualize_cluster_in_grid(
        meshes=meshes,
        cluster_labels=cluster_labels,
        subfolder_names=subfolder_names,
        selected_cluster=left_int,
        num_columns=3,    # e.g. 3 columns
        spacing=1.0       # adjust spacing as you see fit
    )

    right_fig = visualize_cluster_in_grid(
        meshes=meshes,
        cluster_labels=cluster_labels,
        subfolder_names=subfolder_names,
        selected_cluster=right_int,
        num_columns=3,    # e.g. 3 columns
        spacing=1.0
    )

    return left_fig, right_fig


if __name__ == '__main__':
    app.run_server(debug=True)
