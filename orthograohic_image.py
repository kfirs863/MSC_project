
import matplotlib.pyplot as plt

from utils import *


def capture_textured_image_and_depth_from_obj(obj_path):
    rotated_mesh, rotation_matrix = preprocess_mesh(obj_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(rotated_mesh)

    ctr = vis.get_view_control()
    bounds = rotated_mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    camera_distance = max(extent) * 1.5

    lookat = center
    front = [0, 0, -1]
    up = [0, -1, 0]

    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(pinhole_parameters.extrinsic)
    extrinsic[:3, 3] = -np.array([0, 0, camera_distance])
    pinhole_parameters.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(pinhole_parameters)

    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.3)

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

    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    output_image_path, output_depth_path, output_params_path = save_image_and_params(image_np, depth_np, obj_stem,
                                                                                     rotation_matrix, camera_intrinsics,
                                                                                     extrinsic)

    return output_image_path, output_depth_path, output_params_path


if __name__ == '__main__':
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/inscription on Staircase left-20240801T223259Z-001/inscription on Staircase left/stone.obj'
    image_path, camera_params_path = capture_textured_image_and_depth_from_obj(obj_path)
    print(f"Image saved to: {image_path}")
    print(f"Camera intrinsics saved: {camera_params_path}")

    # Load and display the image
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
