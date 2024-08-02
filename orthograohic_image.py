import os
import open3d as o3d
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

from utils import *

def capture_textured_image_from_obj(obj_path):
    rotated_mesh, rotation_matrix = preprocess_mesh(obj_path)

    # Create a visualization window with higher resolution
    window_width = 1920  # Increase the width
    window_height = 1080  # Increase the height
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=window_width, height=window_height)
    vis.add_geometry(rotated_mesh)

    # Adjust the viewpoint to view the wall surface frontally
    ctr = vis.get_view_control()

    # Compute the center and extent of the bounding box
    bounds = rotated_mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()  # This is the center point of the object's bounding box
    extent = bounds.get_extent()  # This gives the dimensions (width, height, depth) of the bounding box

    # Calculate a suitable camera distance to ensure the object fits within the view
    camera_distance = max(extent) * 1.5 # Adjust this factor to ensure the object fits comfortably within the view

    # Ensure the camera is looking directly at the front of the wall
    lookat = center  # The point the camera is looking at, which is the center of the object
    front = [0, 0, -1]  # The direction vector for the camera to face the object directly
    up = [0, -1, 0]  # The up direction for the camera

    # Set the camera parameters using a new set_pinhole_camera_parameters approach
    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(pinhole_parameters.extrinsic)  # Create a copy of the extrinsic matrix
    extrinsic[:3, 3] = -np.array([0, 0, camera_distance])  # Set the camera distance
    pinhole_parameters.extrinsic = extrinsic  # Set the modified extrinsic matrix back to the camera parameters
    ctr.convert_from_pinhole_camera_parameters(pinhole_parameters)

    # Set the viewpoint
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.3)  # Reset the zoom level to 1.0

    # Extract intrinsic parameters of the camera
    intrinsic = pinhole_parameters.intrinsic
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    # Save camera intrinsic parameters
    camera_intrinsics = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }

    # Update the renderer
    vis.poll_events()
    vis.update_renderer()

    # Capture the screen buffer and convert it to an image
    image = vis.capture_screen_float_buffer(True)
    vis.destroy_window()

    # Convert to numpy array
    image_np = (np.asarray(image) * 255).astype(np.uint8)

    # Create the ./images directory if it doesn't exist
    os.makedirs("./images", exist_ok=True)

    # Generate the output file path
    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    output_image_path = f"./images/{obj_stem}_ortho.png"

    # Save the image
    cv2.imwrite(output_image_path, image_np)

    # Save the camera intrinsic parameters and the rotation matrix
    output_params_path = f"./images/{obj_stem}_params.json"
    params = {
        'camera_intrinsics': camera_intrinsics,
        'rotation_matrix': rotation_matrix.tolist()
    }
    with open(output_params_path, 'w') as f:
        json.dump(params, f, indent=4)

    return output_image_path, rotation_matrix, camera_intrinsics

if __name__ == '__main__':
    obj_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/models/inscription on Staircase left-20240801T223259Z-001/inscription on Staircase left/stone.obj'
    image_path, rotation_matrix, camera_intrinsics = capture_textured_image_from_obj(obj_path)
    print(f"Image saved to: {image_path}")
    print(f"Camera intrinsics saved: {camera_intrinsics}")

    # Load and display the image
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
