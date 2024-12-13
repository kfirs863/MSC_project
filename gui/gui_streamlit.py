import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import gc
import plotly.graph_objects as go
import open3d as o3d
from ultralytics import SAM
import os
import json
from tqdm import tqdm  # Ensure tqdm is installed: pip install tqdm
from streamlit_drawable_canvas import st_canvas

# Tools: Adjust these imports if needed
from tools.orthograohic_image import capture_textured_image_and_depth_from_obj
from tools.extract_and_save_masked_areas import extract_and_save_masked_areas
from tools.project_masks_to_3d import project_masks_to_mesh

# ----------------------------
# Initialize Session State
# ----------------------------
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'obj_file' not in st.session_state:
    st.session_state.obj_file = None
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
if 'sam_model_loaded' not in st.session_state:
    st.session_state.sam_model_loaded = False
if 'sam_processed' not in st.session_state:
    st.session_state.sam_processed = False
if 'colored_mesh_path' not in st.session_state:
    st.session_state.colored_mesh_path = None
if 'masks_folders_list' not in st.session_state:
    st.session_state.masks_folders_list = []
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'all_masks' not in st.session_state:
    st.session_state.all_masks = []   # Store boolean masks here
if 'mask_colors' not in st.session_state:
    st.session_state.mask_colors = []

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="3D Mesh Segmentation Tool", layout="wide")

# ----------------------------
# Title and Instructions
# ----------------------------
st.title("3D Mesh Segmentation Tool")
st.markdown("""
**Fix**: We composite the strokes layer over the original mask image
so that the final mask includes user edits, rather than an empty image.
""")

# ----------------------------
# Sidebar Configurations
# ----------------------------
st.sidebar.header("Upload OBJ File")
uploaded_obj_file = st.sidebar.file_uploader("Upload .obj File", type=["obj"])

# If new file is uploaded, reset
if uploaded_obj_file is not None:
    if 'obj_file_name' not in st.session_state or st.session_state.obj_file_name != uploaded_obj_file.name:
        st.session_state.obj_file = uploaded_obj_file
        st.session_state.obj_file_name = uploaded_obj_file.name
        st.session_state.processed = False
        st.session_state.sam_processed = False

st.sidebar.header("Settings")

new_zoom_factor = st.sidebar.slider(
    "Zoom Factor",
    0.1,
    1.0,
    st.session_state.params.get('zoom_factor', st.session_state.zoom_factor),
    0.1
)

show_3d = st.sidebar.checkbox(
    "Show 3D Visualization",
    value=st.session_state.params.get('show_3d', True)
)

st.sidebar.header("Capture Configuration")
number_of_iterations = st.sidebar.slider(
    "Number of Iterations",
    0,
    10,
    st.session_state.params.get('number_of_iterations', 1)
)
use_sharpen = st.sidebar.checkbox(
    "Use Sharpen Filter",
    value=st.session_state.params.get('use_sharpen', False)
)
strength = st.sidebar.slider(
    "Sharpen Strength",
    0.01,
    1.0,
    st.session_state.params.get('strength', 0.1)
)
disable_reflection = st.sidebar.checkbox(
    "Disable Reflection",
    value=st.session_state.params.get('disable_reflection', False)
)
with_texture = st.sidebar.checkbox(
    "With Texture",
    value=st.session_state.params.get('with_texture', True)
)

params_changed = False
if new_zoom_factor != st.session_state.params.get('zoom_factor', None):
    st.session_state.params['zoom_factor'] = new_zoom_factor
    params_changed = True
if show_3d != st.session_state.params.get('show_3d', None):
    st.session_state.params['show_3d'] = show_3d
    params_changed = True
if number_of_iterations != st.session_state.params.get('number_of_iterations', None):
    st.session_state.params['number_of_iterations'] = number_of_iterations
    params_changed = True
if use_sharpen != st.session_state.params.get('use_sharpen', None):
    st.session_state.params['use_sharpen'] = use_sharpen
    params_changed = True
if strength != st.session_state.params.get('strength', None):
    st.session_state.params['strength'] = strength
    params_changed = True
if disable_reflection != st.session_state.params.get('disable_reflection', None):
    st.session_state.params['disable_reflection'] = disable_reflection
    params_changed = True
if with_texture != st.session_state.params.get('with_texture', None):
    st.session_state.params['with_texture'] = with_texture
    params_changed = True

if params_changed:
    st.session_state.processed = False
    st.session_state.sam_processed = False

# ----------------------------
# Function Definitions
# ----------------------------
@st.cache_resource
def load_sam_model(model_path="sam2_l.pt", device="cpu"):
    try:
        model = SAM(model_path)
        model.to(torch.device(device))
        return model
    except Exception as e:
        st.error(f"Failed to load SAM model: {e}")
        return None

def resize_image(image_path, max_size=(1024, 1024)):
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def process_obj(uploaded_file, params):
    tmpdirname = tempfile.mkdtemp()
    temp_obj_path = Path(tmpdirname) / "temp.obj"
    temp_obj_path.write_bytes(uploaded_file.getvalue())

    try:
        image_path, depth_path, camera_params_path = capture_textured_image_and_depth_from_obj(
            str(temp_obj_path),
            zoom=params['zoom_factor'],
            number_of_iterations=params['number_of_iterations'],
            use_sharpen=params['use_sharpen'],
            strength=params['strength'],
            disable_reflection=params['disable_reflection'],
            with_texture=params['with_texture']
        )
        resized_image = resize_image(image_path)
        resized_image_path = Path(tmpdirname) / "resized_image.jpg"
        resized_image.save(resized_image_path, format="JPEG")
        return temp_obj_path, str(resized_image_path), depth_path, camera_params_path
    except Exception as e:
        st.error(f"Failed to process OBJ file: {e}")
        return None, None, None, None

# ----------------------------
# Main Application Logic
# ----------------------------
if not st.session_state.sam_model_loaded:
    with st.spinner("Loading SAM model..."):
        predictor = load_sam_model()
        st.session_state.predictor = predictor
        st.session_state.sam_model_loaded = True
        st.success("SAM model loaded successfully!")
else:
    predictor = st.session_state.predictor

if st.session_state.obj_file is not None:
    if not st.session_state.processed:
        with st.spinner("Processing OBJ file..."):
            temp_obj_path, image_path, depth_path, camera_params_path = process_obj(
                st.session_state.obj_file,
                st.session_state.params
            )
            if all([temp_obj_path, image_path, depth_path, camera_params_path]):
                st.session_state.obj_path = str(temp_obj_path)
                st.session_state.image_path = image_path
                st.session_state.depth_path = depth_path
                st.session_state.camera_params_path = camera_params_path
                st.session_state.processed = True
                st.session_state.sam_processed = False
                gc.collect()
                st.success("OBJ file processed successfully!")
            else:
                st.error("Failed to process the OBJ file.")
                st.stop()

    if st.session_state.processed and st.session_state.image_path:
        try:
            ortho_image = Image.open(st.session_state.image_path).convert("RGB")
            width, height = ortho_image.size
            new_width = int(width * st.session_state.params['zoom_factor'])
            new_height = int(height * st.session_state.params['zoom_factor'])
            scaled_image = ortho_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            st.markdown("### Draw or Edit Bounding Boxes")
            drawing_mode_option = st.selectbox("Drawing tool:", ("Add boxes", "Edit boxes"))
            drawing_mode = "rect" if drawing_mode_option == "Add boxes" else "transform"

            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=scaled_image,
                update_streamlit=True,
                height=new_height,
                width=new_width,
                drawing_mode=drawing_mode,
                key='canvas_bboxes'
            )

            # Extract bounding boxes
            bounding_boxes = []
            if canvas_result.json_data is not None:
                for obj in canvas_result.json_data['objects']:
                    if obj['type'] == 'rect':
                        x1 = obj['left'] / st.session_state.params['zoom_factor']
                        y1 = obj['top'] / st.session_state.params['zoom_factor']
                        x2 = (obj['left'] + obj['width']) / st.session_state.params['zoom_factor']
                        y2 = (obj['top'] + obj['height']) / st.session_state.params['zoom_factor']
                        bounding_boxes.append([x1, y1, x2, y2])

                if len(bounding_boxes) > 10:
                    st.warning("Max 10 bounding boxes allowed.")
                    bounding_boxes = bounding_boxes[:10]
            else:
                st.info("No bounding boxes added yet.")

            # Process bounding boxes with SAM
            if st.button("Process with SAM"):
                if not bounding_boxes:
                    st.warning("Please draw at least one bounding box.")
                else:
                    if predictor is None:
                        st.error("SAM model not loaded.")
                    else:
                        with st.spinner("Processing bounding boxes with SAM..."):
                            try:
                                bboxes = np.array(bounding_boxes)
                                results = predictor(
                                    source=st.session_state.image_path,
                                    bboxes=bboxes,
                                    conf=0.5,
                                    retina_masks=True
                                )

                                st.session_state.all_masks = []
                                st.session_state.mask_colors = []

                                all_masks = []
                                for result in results:
                                    if result.masks is not None:
                                        for mask in result.masks.data:
                                            # Convert to boolean
                                            mask_np = mask.cpu().numpy().astype(bool)
                                            mask_np = np.squeeze(mask_np)
                                            if mask_np.ndim != 2:
                                                st.warning(f"Skipping invalid mask shape {mask_np.shape}")
                                                continue
                                            all_masks.append(mask_np)

                                if not all_masks:
                                    st.error("No valid masks generated.")
                                    st.stop()

                                st.session_state.all_masks = all_masks
                                st.session_state.mask_colors = np.random.randint(100, 255, (len(all_masks), 3))
                                st.session_state.sam_processed = True

                                st.markdown("### Generated Masks")
                                cols_per_row = 3
                                num_masks = len(all_masks)
                                num_rows = (num_masks + cols_per_row - 1) // cols_per_row

                                mask_idx = 0
                                for row in range(num_rows):
                                    cols = st.columns(cols_per_row)
                                    for col in cols:
                                        if mask_idx < num_masks:
                                            boolean_mask = all_masks[mask_idx]
                                            # Convert boolean to black/white
                                            mask_display = np.where(boolean_mask, 255, 0).astype(np.uint8)
                                            mask_pil = Image.fromarray(mask_display)
                                            resized_mask = mask_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                            with col:
                                                st.image(resized_mask, caption=f"Mask {mask_idx+1}", width=new_width)
                                            mask_idx += 1

                                st.success("Masks generated successfully!")
                                gc.collect()

                            except Exception as e:
                                st.error(f"SAM processing error: {e}")

            # Single editing widget + dropbox to switch masks
            if st.session_state.sam_processed and st.session_state.all_masks:
                st.markdown("### Optional: Edit Masks (Composite Fix)")
                mask_options = [f"Mask {i+1}" for i in range(len(st.session_state.all_masks))]
                mask_to_edit = st.selectbox("Select a mask to edit:", options=mask_options)
                selected_mask_idx = int(mask_to_edit.split()[-1]) - 1

                current_mask_bool = st.session_state.all_masks[selected_mask_idx]
                if current_mask_bool.size == 0 or current_mask_bool.ndim != 2:
                    st.warning(f"Invalid shape {current_mask_bool.shape} for mask {selected_mask_idx+1}")
                else:
                    # Convert boolean => black/white RGBA
                    # White = True (255), Black = False (0)
                    bw_display = np.where(current_mask_bool, 255, 0).astype(np.uint8)
                    mask_pil = Image.fromarray(bw_display, mode="L").convert("RGBA")

                    st.markdown("**Brush**: White (Add) or Black (Erase). The final mask is a composite of strokes + original mask.")
                    brush_option = st.radio("Brush color:", ["White (Add)", "Black (Erase)"])
                    stroke_color = "rgba(255,255,255,1.0)" if brush_option == "White (Add)" else "rgba(0,0,0,1.0)"
                    stroke_width = st.slider("Brush size", 1, 50, 10)

                    edit_canvas_result = st_canvas(
                        fill_color="#00000000",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_image=mask_pil,  # The mask is behind the canvas
                        update_streamlit=True,
                        height=mask_pil.height,
                        width=mask_pil.width,
                        drawing_mode="freedraw",
                        key='mask_editor'
                    )

                    if st.button("Save Mask Edits"):
                        if edit_canvas_result.image_data is not None:
                            # The returned image_data is ONLY the strokes on transparent
                            strokes_rgba = edit_canvas_result.image_data  # shape (H, W, 4)

                            # Composite strokes over original mask image
                            original_rgba = mask_pil.copy()  # The original mask as RGBA
                            strokes_pil = Image.fromarray(strokes_rgba.astype(np.uint8), mode="RGBA")

                            # Composite user strokes (foreground) over mask background
                            # White strokes will set R/G/B=255, alpha=255
                            # Black strokes => R/G/B=0, alpha=255
                            combined = Image.alpha_composite(original_rgba, strokes_pil)

                            # Convert combined back to boolean mask by thresholding
                            # E.g., treat 'white (255,255,255)' as True, black(0,0,0) as False
                            combined_np = np.array(combined)
                            # A simple approach is to check alpha or grayscale
                            # If alpha > 128 => keep old or painted white
                            # But let's do grayscale of the RGB channels
                            rgb_gray = np.mean(combined_np[:, :, :3], axis=2)
                            refined_mask = (rgb_gray > 128)

                            st.session_state.all_masks[selected_mask_idx] = refined_mask
                            st.success(f"Edits saved for {mask_to_edit}!")

            # Projection to 3D
            if st.session_state.sam_processed and st.session_state.all_masks:
                if st.button("Project Masks to 3D Mesh"):
                    st.info("Projecting masks onto the 3D mesh...")
                    try:
                        # Convert final masks to boolean
                        all_masks_bool = np.array(st.session_state.all_masks).astype(bool)
                        mask_colors = np.array(st.session_state.mask_colors)

                        image_path_obj = Path(st.session_state.image_path)
                        masks_path = image_path_obj.parent / f"{image_path_obj.stem}_masks.npy"
                        colors_path = image_path_obj.parent / f"{image_path_obj.stem}_colors.npy"

                        np.save(masks_path, all_masks_bool)
                        np.save(colors_path, mask_colors)

                        colored_mesh_path = project_masks_to_mesh(
                            obj_path=st.session_state.obj_path,
                            masks_path=str(masks_path),
                            colors_path=str(colors_path),
                            params_path=st.session_state.camera_params_path,
                            depth_path=st.session_state.depth_path
                        )

                        st.info("Extracting and saving masked areas...")
                        output_dir = Path('datasets') / Path(st.session_state.obj_path).stem
                        masks_folders_list = extract_and_save_masked_areas(colored_mesh_path, output_dir, "mask")

                        st.session_state.colored_mesh_path = colored_mesh_path
                        st.session_state.masks_folders_list = masks_folders_list
                        st.success("Projection & extraction completed successfully!")

                        # 3D Visualization
                        if show_3d:
                            st.markdown("### 3D Visualization")
                            st.info("Rendering 3D visualizations...")
                            col1, col2 = st.columns(2)

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
                                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                                    ),
                                    margin=dict(l=0, r=0, t=0, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)

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
                                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                                            ),
                                            margin=dict(l=0, r=0, t=0, b=0)
                                        )
                                        st.markdown(f"Masked Area {idx+1}")
                                        st.plotly_chart(fig, use_container_width=True)

                            st.success("3D visualization rendered successfully!")

                        # Download buttons
                        st.markdown("### Download Meshes")
                        with open(st.session_state.colored_mesh_path, "rb") as f:
                            st.download_button(
                                "Download Complete Colored Mesh (.ply)",
                                f,
                                file_name=Path(st.session_state.colored_mesh_path).name,
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

                    except Exception as e:
                        st.error(f"Projection error: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.exception(e)
else:
    st.info("Please upload an `.obj` file to begin.")
