import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile
import gc
import plotly.graph_objects as go
import open3d as o3d
from ultralytics import SAM

# Import your custom tools/modules here
# These should be defined in your project
from tools.utils import display_masked_areas
from tools.orthograohic_image import capture_textured_image_and_depth_from_obj
from tools.extract_and_save_masked_areas import extract_and_save_masked_areas
from tools.project_masks_to_3d import project_masks_to_mesh

from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Initialize Session State
# ----------------------------
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'bounding_boxes' not in st.session_state:
    st.session_state.bounding_boxes = []
if 'obj_path' not in st.session_state:
    st.session_state.obj_path = None
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 0.4
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'depth_path' not in st.session_state:
    st.session_state.depth_path = None
if 'camera_params_path' not in st.session_state:
    st.session_state.camera_params_path = None
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0  # Used to reset the canvas

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="3D Mesh Segmentation Tool", layout="wide")

# ----------------------------
# Title and Instructions
# ----------------------------
st.title("3D Mesh Segmentation Tool")
st.markdown("""
Upload an `.obj` file, draw multiple bounding boxes on the generated image, and process them with SAM.
""")

# ----------------------------
# Handle Deprecation Warning
# ----------------------------
# Ensure that `use_container_width` is used instead of `use_column_width`
# All instances of `use_column_width=True` should be replaced with `use_container_width=True`

# ----------------------------
# Function Definitions
# ----------------------------

def load_sam_model(model_path="sam2_l.pt", device="cpu"):
    """
    Load the SAM model.
    """
    try:
        model = SAM(model_path)
        model.to(torch.device(device))
        return model
    except Exception as e:
        st.error(f"Failed to load SAM model: {e}")
        return None

def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resize the image to fit within max_size while maintaining aspect ratio.
    """
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def process_obj(uploaded_file, zoom, number_of_iterations, use_sharpen, strength, disable_reflection, with_texture):
    """
    Process the uploaded OBJ file to generate a textured image and other parameters.
    """
    tmpdirname = tempfile.mkdtemp()
    temp_obj_path = Path(tmpdirname) / "temp.obj"
    temp_obj_path.write_bytes(uploaded_file.getvalue())

    try:
        image_path, depth_path, camera_params_path = capture_textured_image_and_depth_from_obj(
            str(temp_obj_path),
            zoom=zoom,
            number_of_iterations=number_of_iterations,
            use_sharpen=use_sharpen,
            strength=strength,
            disable_reflection=disable_reflection,
            with_texture=with_texture
        )
        resized_image = resize_image(image_path)
        resized_image_path = Path(tmpdirname) / "resized_image.jpg"
        resized_image.save(resized_image_path, format="JPEG")
        return temp_obj_path, str(resized_image_path), depth_path, camera_params_path
    except Exception as e:
        st.error(f"Failed to process OBJ file: {e}")
        return None, None, None, None

def draw_bounding_boxes(image, bounding_boxes, zoom_factor):
    """
    Draw all bounding boxes on the image.
    """
    draw = ImageDraw.Draw(image)
    for bbox in bounding_boxes:
        # Scale bounding box coordinates based on zoom_factor
        scaled_bbox = [coord * zoom_factor for coord in bbox]
        draw.rectangle(scaled_bbox, outline="red", width=2)
    return image

def remove_bbox(idx):
    """
    Remove a bounding box from the session state.
    """
    if 0 <= idx < len(st.session_state.bounding_boxes):
        removed_box = st.session_state.bounding_boxes.pop(idx)
        st.success(f"Removed Box {idx + 1}: {removed_box}")
        st.session_state.canvas_key += 1  # Reset the canvas to update the background

# ----------------------------
# Load SAM Model
# ----------------------------
with st.spinner("Loading SAM model..."):
    predictor = load_sam_model()

# ----------------------------
# Sidebar Configurations
# ----------------------------
st.sidebar.header("Settings")
new_zoom_factor = st.sidebar.slider("Zoom Factor", 0.1, 1.0, st.session_state.zoom_factor, 0.1)
show_3d = st.sidebar.checkbox("Show 3D Visualization", value=False)

# Capture Configuration Settings
st.sidebar.header("Capture Configuration")
zoom = st.sidebar.slider("Zoom Level", 0.1, 2.0, 1.0, 0.1)
number_of_iterations = st.sidebar.slider("Number of Iterations", 0, 10, 1)
use_sharpen = st.sidebar.checkbox("Use Sharpen Filter", value=True)
strength = st.sidebar.slider("Sharpen Strength", 0.01, 1.0, 0.1)
disable_reflection = st.sidebar.checkbox("Disable Reflection", value=False)
with_texture = st.sidebar.checkbox("With Texture", value=True)

# Update zoom factor in session state if changed
if new_zoom_factor != st.session_state.zoom_factor:
    st.session_state.zoom_factor = new_zoom_factor
    st.session_state.processed = False

# ----------------------------
# File Uploader for .obj Files
# ----------------------------
obj_file = st.sidebar.file_uploader("Upload .obj File", type=["obj"])

# ----------------------------
# Main Application Logic
# ----------------------------
if obj_file is not None and not st.session_state.processed:
    with st.spinner("Processing OBJ file..."):
        temp_obj_path, image_path, depth_path, camera_params_path = process_obj(
            obj_file,
            zoom=st.session_state.zoom_factor,
            number_of_iterations=number_of_iterations,
            use_sharpen=use_sharpen,
            strength=strength,
            disable_reflection=disable_reflection,
            with_texture=with_texture
        )
        if all([temp_obj_path, image_path, depth_path, camera_params_path]):
            st.session_state.obj_path = str(temp_obj_path)
            st.session_state.image_path = image_path
            st.session_state.depth_path = depth_path
            st.session_state.camera_params_path = camera_params_path
            st.session_state.processed = True
            gc.collect()

# If the OBJ file has been processed, display the image and bounding box tools
if st.session_state.processed and st.session_state.image_path:
    try:
        # Load the processed image
        ortho_image = Image.open(st.session_state.image_path).convert("RGB")
        width, height = ortho_image.size
        new_width = int(width * st.session_state.zoom_factor)
        new_height = int(height * st.session_state.zoom_factor)
        scaled_image = ortho_image.resize((new_width, new_height), Image.LANCZOS)

        # Draw existing bounding boxes on the image
        image_with_boxes = ortho_image.copy()
        if st.session_state.bounding_boxes:
            image_with_boxes = draw_bounding_boxes(image_with_boxes, st.session_state.bounding_boxes, st.session_state.zoom_factor)
        resized_image_with_boxes = image_with_boxes.resize((new_width, new_height), Image.LANCZOS)

        st.markdown("### Draw Bounding Boxes")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent fill
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=resized_image_with_boxes,
            update_streamlit=True,
            height=new_height,
            width=new_width,
            drawing_mode="rect",
            key=f"canvas_{st.session_state.canvas_key}",  # Unique key to reset the canvas
        )

        # Handle bounding box drawing
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                # Iterate through all drawn objects
                for obj in objects:
                    if obj["type"] == "rect":
                        # Extract bounding box coordinates
                        x1 = obj["left"] / st.session_state.zoom_factor
                        y1 = obj["top"] / st.session_state.zoom_factor
                        x2 = (obj["left"] + obj["width"]) / st.session_state.zoom_factor
                        y2 = (obj["top"] + obj["height"]) / st.session_state.zoom_factor
                        bbox = [x1, y1, x2, y2]

                        # Add the new bounding box to the session state list
                        if len(st.session_state.bounding_boxes) < 10:
                            # Check if the bbox is already in the list to prevent duplicates
                            if bbox not in st.session_state.bounding_boxes:
                                st.session_state.bounding_boxes.append(bbox)
                                st.success(f"Added Box {len(st.session_state.bounding_boxes)}: {bbox}")
                                st.session_state.canvas_key += 1  # Reset the canvas after adding a box
                            else:
                                st.warning("This bounding box already exists.")
                        else:
                            st.warning("Maximum 10 bounding boxes allowed.")

        # Display existing bounding boxes and provide removal options
        st.write("### Bounding Boxes:")
        if st.session_state.bounding_boxes:
            # Create a selectbox to choose which bounding box to remove
            box_labels = [
                f"Box {idx + 1}: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
                for idx, bbox in enumerate(st.session_state.bounding_boxes)
            ]
            selected_box_label = st.selectbox("Select a bounding box to remove:", box_labels)

            # Find the index of the selected box
            selected_idx = box_labels.index(selected_box_label)

            # Button to remove the selected bounding box
            if st.button("Remove Selected Box"):
                remove_bbox(selected_idx)

        else:
            st.info("No bounding boxes added yet.")

        # Button to process the bounding boxes with SAM
        if st.button("Process with SAM"):
            if st.session_state.bounding_boxes:
                if predictor is None:
                    st.error("SAM model not loaded.")
                else:
                    with st.spinner("Processing with SAM..."):
                        try:
                            # Convert bounding boxes to NumPy array
                            bboxes = np.array(st.session_state.bounding_boxes)
                            # Process with SAM
                            results = predictor(
                                source=st.session_state.image_path,
                                bboxes=bboxes,
                                conf=0.5,
                                retina_masks=True
                            )

                            all_masks = []
                            st.markdown("### Generated Masks")

                            for result in results:
                                if result.masks is not None:
                                    for mask in result.masks.data:
                                        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                                        all_masks.append(mask_np)
                                        mask_display = (mask_np * 255).astype(np.uint8)
                                        st.image(mask_display, caption=f'Mask {len(all_masks)}', use_container_width=True)

                            if not all_masks:
                                st.error("No masks generated.")
                                st.stop()

                            # Save masks and colors
                            masks_np = np.array(all_masks)
                            mask_colors = np.random.randint(100, 255, (len(masks_np), 3))

                            image_path_obj = Path(st.session_state.image_path)
                            masks_path = image_path_obj.parent / f"{image_path_obj.stem}_masks.npy"
                            colors_path = image_path_obj.parent / f"{image_path_obj.stem}_colors.npy"

                            np.save(masks_path, masks_np)
                            np.save(colors_path, mask_colors)

                            # Project masks to mesh
                            colored_mesh_path = project_masks_to_mesh(
                                obj_path=st.session_state.obj_path,
                                masks_path=str(masks_path),
                                colors_path=str(colors_path),
                                params_path=st.session_state.camera_params_path,
                                depth_path=st.session_state.depth_path
                            )

                            # Extract and save masked areas
                            output_dir = Path('datasets') / Path(st.session_state.obj_path).stem
                            masks_folders_list = extract_and_save_masked_areas(colored_mesh_path, output_dir, "mask")

                            st.markdown("### Download Meshes")
                            with open(colored_mesh_path, "rb") as f:
                                st.download_button(
                                    "Download Complete Colored Mesh (.ply)",
                                    f,
                                    file_name=Path(colored_mesh_path).name,
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )

                            for masked_area_folder in masks_folders_list:
                                mesh_path = masked_area_folder / 'mask.ply'
                                if mesh_path.exists():
                                    with open(mesh_path, "rb") as f:
                                        st.download_button(
                                            f"Download {masked_area_folder.name}",
                                            f,
                                            file_name=f"{masked_area_folder.name}.ply",
                                            mime="application/octet-stream",
                                            key=f"download_{masked_area_folder.name}",
                                            use_container_width=True
                                        )

                            # 3D Visualization (Optional)
                            if show_3d:
                                st.markdown("### 3D Visualization")
                                col1, col2 = st.columns(2)

                                # Complete colored mesh
                                with col1:
                                    st.markdown("#### Complete Mesh")
                                    colored_mesh = o3d.io.read_triangle_mesh(colored_mesh_path)
                                    vertices = np.asarray(colored_mesh.vertices)
                                    triangles = np.asarray(colored_mesh.triangles)
                                    colors = np.asarray(colored_mesh.vertex_colors)

                                    fig = go.Figure(data=[
                                        go.Mesh3d(
                                            x=vertices[:, 0],
                                            y=vertices[:, 1],
                                            z=vertices[:, 2],
                                            i=triangles[:, 0],
                                            j=triangles[:, 1],
                                            k=triangles[:, 2],
                                            vertexcolor=colors,
                                            opacity=1.0
                                        )
                                    ])

                                    fig.update_layout(
                                        scene=dict(
                                            aspectmode='data',
                                            camera=dict(
                                                eye=dict(x=1.5, y=1.5, z=1.5)
                                            )
                                        ),
                                        margin=dict(l=0, r=0, t=0, b=0)
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                # Individual masked areas
                                with col2:
                                    st.markdown("#### Individual Masked Areas")
                                    for idx, masked_area_folder in enumerate(masks_folders_list):
                                        mesh_path = masked_area_folder / 'mask.ply'
                                        if mesh_path.exists():
                                            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                                            vertices = np.asarray(mesh.vertices)
                                            triangles = np.asarray(mesh.triangles)
                                            colors = np.asarray(mesh.vertex_colors)

                                            fig = go.Figure(data=[
                                                go.Mesh3d(
                                                    x=vertices[:, 0],
                                                    y=vertices[:, 1],
                                                    z=vertices[:, 2],
                                                    i=triangles[:, 0],
                                                    j=triangles[:, 1],
                                                    k=triangles[:, 2],
                                                    vertexcolor=colors,
                                                    opacity=1.0
                                                )
                                            ])

                                            fig.update_layout(
                                                scene=dict(
                                                    aspectmode='data',
                                                    camera=dict(
                                                        eye=dict(x=1.5, y=1.5, z=1.5)
                                                    )
                                                ),
                                                margin=dict(l=0, r=0, t=0, b=0)
                                            )

                                            st.markdown(f"Masked Area {idx + 1}")
                                            st.plotly_chart(fig, use_container_width=True)

                            # Clean up memory
                            del all_masks, masks_np, mask_colors
                            gc.collect()

                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
            else:
                st.warning("Please draw at least one bounding box.")

    except Exception as e:
        st.error(f"Error loading image: {e}")
else:
    st.info("Please upload an `.obj` file to begin.")
