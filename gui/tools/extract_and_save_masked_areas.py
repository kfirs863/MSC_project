from pathlib import Path

import open3d as o3d
import numpy as np
import os


def extract_and_save_masked_areas(input_ply_path, output_dir, label) -> list:
    output_dir_list = []
    # Load the colored mesh
    colored_masks_mesh = o3d.io.read_triangle_mesh(input_ply_path)

    # Ensure the mesh has vertex colors
    if not colored_masks_mesh.has_vertex_colors():
        raise ValueError("The mesh does not have vertex colors.")

    # Convert mesh to numpy arrays
    vertices = np.asarray(colored_masks_mesh.vertices)
    vertex_colors = np.asarray(colored_masks_mesh.vertex_colors)
    triangles = np.asarray(colored_masks_mesh.triangles)

    # Identify unique colors (ignore white [1, 1, 1])
    unique_colors = np.unique(vertex_colors, axis=0)
    unique_colors = unique_colors[np.any(unique_colors != [1, 1, 1], axis=1)]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each unique color
    for i, color in enumerate(unique_colors):
        # Filter vertices matching the current color
        color_mask = np.all(vertex_colors == color, axis=1)
        filtered_vertex_indices = np.where(color_mask)[0]

        # Map original vertex indices to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_vertex_indices)}

        # Find triangles (faces) that have all vertices with the current color
        mask_faces = []
        for tri in triangles:
            if all(v_idx in filtered_vertex_indices for v_idx in tri):
                # Remap indices to new ones
                remapped_face = [index_map[v_idx] for v_idx in tri]
                mask_faces.append(remapped_face)

        mask_faces = np.array(mask_faces)

        # If no faces were found, skip this color
        if len(mask_faces) == 0:
            continue

        # Create a new mesh for the masked area
        masked_mesh = o3d.geometry.TriangleMesh()
        masked_mesh.vertices = o3d.utility.Vector3dVector(vertices[filtered_vertex_indices])
        masked_mesh.triangles = o3d.utility.Vector3iVector(mask_faces)
        masked_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[filtered_vertex_indices])

        # Save the masked area mesh to a new file
        output_ply_path = Path(output_dir, f"{label}_mask_{i}")
        output_ply_path.mkdir(parents=True, exist_ok=True)
        output_dir_list.append(output_ply_path)

        output_ply_path = output_ply_path / 'mask.ply'
        o3d.io.write_triangle_mesh(output_ply_path.as_posix(), masked_mesh)

        print(f"Saved masked area {i} with color {color} to {output_ply_path}")

    print("Finished extracting and saving all masked areas.")
    return output_dir_list
