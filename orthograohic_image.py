
import matplotlib.pyplot as plt

from utils import *


def capture_textured_image_and_depth_from_obj(obj_path, zoom=1.0,number_of_iterations=1,strength=1):
    rotated_mesh, rotation_matrix = preprocess_mesh(obj_path)

    # Visualize the mesh
    # o3d.visualization.draw_geometries([rotated_mesh])
    if number_of_iterations > 0:
        rotated_mesh = rotated_mesh.filter_sharpen(number_of_iterations=number_of_iterations, strength=strength)
        rotated_mesh.compute_vertex_normals()

    # Color the mesh by vertex normals
    # color_mesh_by_vertex_normals(rotated_mesh)
    resolution = 1024
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=resolution, height=resolution)
    vis.add_geometry(rotated_mesh)

    ctr = vis.get_view_control()
    bounds = rotated_mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()


    lookat = center
    front = [0, 0, 1]
    up = [-1,0, 0]

    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
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

    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    output_image_path, output_depth_path, output_params_path = save_image_and_params(image_np, depth_np, obj_stem,
                                                                                     rotation_matrix, camera_intrinsics,
                                                                                     extrinsic)

    return output_image_path, output_depth_path, output_params_path

if __name__ == '__main__':
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/wall 5-20240803T111405Z-001/wall 5/wall 5/5_3D/5_HR.obj'
    # obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/Crosses on Staircase left/staircase_left.obj'
    output_image_path, output_depth_path, output_params_path = capture_textured_image_and_depth_from_obj(obj_path,zoom=0.4,number_of_iterations=0,strength=0)
    print(f"Image saved to: {output_image_path}")
    print(f"Camera intrinsics saved: {output_params_path}")
    # Load and display the image
    image = cv2.imread(output_image_path, cv2.IMREAD_COLOR)
    plt.imshow(image)
    plt.axis('off')

    plt.show()
