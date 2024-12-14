# utils.py

def load_feature_descriptions():
    """Load feature descriptions."""
    feature_descriptions = {
        'surface_area': 'The total area of the mesh surface.',
        'sa_to_bbox_volume_ratio': 'The ratio of surface area to bounding box volume.',
        'mean_curvature': 'The average curvature of the mesh surface.',
        'std_curvature': 'The standard deviation of curvature across the mesh surface.',
        'mean_edge_length': 'The average length of edges in the mesh.',
        'std_edge_length': 'The standard deviation of edge lengths in the mesh.',
        'mean_face_area': 'The average area of faces in the mesh.',
        'std_face_area': 'The standard deviation of face areas in the mesh.',
        'reflection_symmetry_x': 'Reflection symmetry across the X-axis.',
        'reflection_symmetry_y': 'Reflection symmetry across the Y-axis.',
        'reflection_symmetry_z': 'Reflection symmetry across the Z-axis.',
        'curvature_histogram': 'Histogram of curvature values across the mesh surface.',
    }
    return feature_descriptions
