import open3d as o3d
import numpy as np
import os

def extract_and_save_masked_areas(input_ply_path, output_dir):
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
        output_ply_path = os.path.join(output_dir, f"masked_area_{i}.ply")
        o3d.io.write_triangle_mesh(output_ply_path, masked_mesh)

        print(f"Saved masked area {i} with color {color} to {output_ply_path}")

    print("Finished extracting and saving all masked areas.")


def display_masked_areas(output_dir):
    # List all the files in the output directory
    files = os.listdir(output_dir)

    # Load each file and display it
    for i, file in enumerate(files):
        if file.endswith(".ply"):
            # Load the mesh instead of point cloud
            masked_mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, file))
            if masked_mesh.is_empty():
                print(f"Skipped empty mesh file: {file}")
                continue
            masked_mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([masked_mesh], window_name=f"Masked Area {i}")

    print(f"Displayed {len(files)} masked areas.")

if __name__ == '__main__':
    # extract_and_save_masked_areas('/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/notebook/images/3D_herald_colored_mesh.ply', '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/herald-engraving-saint-helena-chapel/source/masked_areas')
    display_masked_areas('/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/herald-engraving-saint-helena-chapel/source/masked_areas')



