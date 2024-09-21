import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, dilation
from skimage.util import invert
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import os

from tools.extract_and_save_masked_areas import extract_and_save_masked_areas
from tools.orthograohic_image import capture_textured_image_and_depth_from_obj
from tools.project_masks_to_3d import project_masks_to_mesh
from tools.utils import display_masked_areas


def process_ply_to_skeleton(folder_path: str,project_on_mesh: bool = False) -> None:

    # Find the .ply file in the given folder
    ply_path = None
    for file in os.listdir(folder_path):
        if file.endswith('.ply'):
            ply_path = os.path.join(folder_path, file)
            break

    # Capture textured image and depth from the PLY file
    output_image_path, output_depth_path, output_params_path = capture_textured_image_and_depth_from_obj(
        ply_path, zoom=0.8, number_of_iterations=0,disable_reflection=True
    )

    # Load the image and convert to grayscale
    image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

    # Enhance contrast using histogram equalization (if needed)
    image = cv2.equalizeHist(image)

    # Apply Otsu's thresholding
    thresh_value = threshold_otsu(image)
    binary_image = image > thresh_value

    # Invert the image to make the object white and background black
    invert_binary_image = invert(binary_image)

    # Skeletonize the binary image
    skeleton = skeletonize(invert_binary_image)

    # Make the skeleton lines thicker by applying dilation
    thicker_skeleton = dilation(skeleton, footprint=np.ones((3, 3)))

    # Save the skeleton image in the same folder as the PLY file
    skeleton_image_path = os.path.join(os.path.dirname(ply_path), 'skeleton.png')
    plt.imsave(skeleton_image_path, thicker_skeleton, cmap='gray')

    print(f"Skeleton image saved at: {skeleton_image_path}")

    if project_on_mesh:
        # Save the skeleton array as a NumPy file
        skeleton_array_path = os.path.join(os.path.dirname(ply_path), 'skeleton_skel_2.npy')
        thicker_skeleton = np.expand_dims(thicker_skeleton, axis=0)
        np.save(skeleton_array_path, thicker_skeleton)

        # Create red color array for the skeleton and save it as a NumPy file
        colors_path = os.path.join(os.path.dirname(ply_path), 'colors.npy')
        colors = np.random.randint(100, 255, (1, 3))

        np.save(colors_path, colors)

        # Project the masks to the mesh
        colored_mesh_path = project_masks_to_mesh(ply_path, skeleton_array_path, colors_path, output_params_path, output_depth_path)

        # Extract the skeleton from the 3D mesh
        extract_and_save_masked_areas(colored_mesh_path, output_dir=os.path.dirname(ply_path), label='skeleton')

if __name__ == '__main__':
    folder_path = '/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets/staircase_left_mask_20'
    process_ply_to_skeleton(folder_path, project_on_mesh=True)
    # display_masked_areas('/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets/staircase_left_mask_20')

