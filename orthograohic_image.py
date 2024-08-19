
import matplotlib.pyplot as plt

from utils import *


def capture_textured_image_and_depth_from_obj(obj_path, zoom_factor=0.5, use_super_resolution=False):
    rotated_mesh, rotation_matrix = preprocess_mesh(obj_path)

    # Visualize the mesh
    o3d.visualization.draw_geometries([rotated_mesh])

    rotated_mesh = rotated_mesh.filter_smooth_simple(number_of_iterations=1)
    rotated_mesh.compute_vertex_normals()

    # Color the mesh by vertex normals
    # color_mesh_by_vertex_normals(rotated_mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=1024)
    vis.add_geometry(rotated_mesh)

    ctr = vis.get_view_control()
    bounds = rotated_mesh.get_axis_aligned_bounding_box()
    center = rotated_mesh.get_center()
    extent = bounds.get_extent()
    camera_distance = max(extent) * 1.5

    lookat = center
    front = [0, 0, 1]
    up = [0, 1, 0]

    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(pinhole_parameters.extrinsic)
    extrinsic[:3, 3] = -np.array([0, 0, camera_distance])

    pinhole_parameters.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(pinhole_parameters)

    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom_factor)

    intrinsic = pinhole_parameters.intrinsic
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    extrinsic = pinhole_parameters.extrinsic

    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(True)
    depth = vis.capture_depth_float_buffer(True)
    vis.destroy_window()

    image_np = (np.asarray(image) * 255).astype(np.uint8)
    depth_np = np.asarray(depth)

    # Enhance the captured image
    image_np = enhance_image(image_np)

    # Apply super-resolution if the flag is set
    if use_super_resolution:
        super_resolution_factor = 2
        image_np = super_resolution(image_np, factor=super_resolution_factor)

        # Update camera intrinsics after super-resolution
        camera_intrinsics['fx'] *= super_resolution_factor
        camera_intrinsics['fy'] *= super_resolution_factor
        camera_intrinsics['cx'] *= super_resolution_factor
        camera_intrinsics['cy'] *= super_resolution_factor

        # Resize the depth map to match the new resolution
        depth_np = cv2.resize(depth_np, (depth_np.shape[1] * super_resolution_factor, depth_np.shape[0] * super_resolution_factor), interpolation=cv2.INTER_LINEAR)

    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    output_image_path, output_depth_path, output_params_path = save_image_and_params(image_np, depth_np, obj_stem,
                                                                                     rotation_matrix, camera_intrinsics,
                                                                                     extrinsic)

    return output_image_path, output_depth_path, output_params_path

if __name__ == '__main__':
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/valid_models/S01/S01.obj'
    # obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/Crosses on Staircase left/staircase_left.obj'
    output_image_path, output_depth_path, output_params_path = capture_textured_image_and_depth_from_obj(obj_path,zoom_factor=0.50)
    print(f"Image saved to: {output_image_path}")
    print(f"Camera intrinsics saved: {output_params_path}")
    # Load and display the image
    image = cv2.imread(output_image_path, cv2.IMREAD_COLOR)
    plt.imshow(image)
    plt.axis('off')

    plt.show()
