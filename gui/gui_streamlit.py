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

# Import your custom tools/modules here
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
# Sidebar Configurations
# ----------------------------
st.sidebar.header("Upload OBJ File")
# Moved the file uploader to the top of the sidebar
uploaded_obj_file = st.sidebar.file_uploader("Upload .obj File", type=["obj"])

# Store the uploaded file in session state
if uploaded_obj_file is not None:
    if 'obj_file_name' not in st.session_state or st.session_state.obj_file_name != uploaded_obj_file.name:
        st.session_state.obj_file = uploaded_obj_file
        st.session_state.obj_file_name = uploaded_obj_file.name
        # Reset processed flags when a new file is uploaded
        st.session_state.processed = False
        st.session_state.sam_processed = False

st.sidebar.header("Settings")

# Zoom Factor
new_zoom_factor = st.sidebar.slider(
    "Zoom Factor",
    0.1,
    1.0,
    st.session_state.params.get('zoom_factor', st.session_state.zoom_factor),
    0.1
)

# Show 3D Visualization
show_3d = st.sidebar.checkbox(
    "Show 3D Visualization",
    value=st.session_state.params.get('show_3d', False)
)

# Capture Configuration Settings
st.sidebar.header("Capture Configuration")
number_of_iterations = st.sidebar.slider(
    "Number of Iterations",
    0,
    10,
    st.session_state.params.get('number_of_iterations', 1)
)
use_sharpen = st.sidebar.checkbox(
    "Use Sharpen Filter",
    value=st.session_state.params.get('use_sharpen', True)
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

# Check if any parameter has changed
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

# If any parameter has changed, reset the processed flags
if params_changed:
    st.session_state.processed = False
    st.session_state.sam_processed = False

# ----------------------------
# Function Definitions
# ----------------------------

@st.cache_resource
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

def process_obj(uploaded_file, params):
    """
    Process the uploaded OBJ file to generate a textured image and other parameters.
    """
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

# Load the SAM model only once
if not st.session_state.sam_model_loaded:
    with st.spinner("Loading SAM model..."):
        st.info("Loading the SAM model. This may take a few moments...")
        predictor = load_sam_model()
        st.session_state.predictor = predictor
        st.session_state.sam_model_loaded = True
        st.success("SAM model loaded successfully!")
else:
    predictor = st.session_state.predictor

if st.session_state.obj_file is not None:
    # Process OBJ file if not already processed
    if not st.session_state.processed:
        with st.spinner("Processing OBJ file..."):
            st.info("Generating textured image and depth map from the OBJ file...")
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
                st.session_state.sam_processed = False  # Reset SAM processed flag
                gc.collect()
                st.success("OBJ file processed successfully!")
            else:
                st.error("Failed to process the OBJ file.")
                st.stop()
    # If the OBJ file has been processed, display the image and bounding box tools
    if st.session_state.processed and st.session_state.image_path:
        try:
            # Load the processed image
            ortho_image = Image.open(st.session_state.image_path).convert("RGB")
            width, height = ortho_image.size
            new_width = int(width * st.session_state.params['zoom_factor'])
            new_height = int(height * st.session_state.params['zoom_factor'])
            scaled_image = ortho_image.resize((new_width, new_height), Image.LANCZOS)

            st.markdown("### Draw or Edit Bounding Boxes")
            drawing_mode_option = st.selectbox(
                "Drawing tool:",
                ("Add boxes", "Edit boxes"),
            )

            if drawing_mode_option == "Add boxes":
                drawing_mode = "rect"
            elif drawing_mode_option == "Edit boxes":
                drawing_mode = "transform"
            else:
                drawing_mode = "rect"

            # Use st_canvas to draw bounding boxes
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent fill
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=scaled_image,
                update_streamlit=True,
                height=new_height,
                width=new_width,
                drawing_mode=drawing_mode,
                key='canvas',
            )

            # Extract bounding boxes from canvas objects
            bounding_boxes = []
            if canvas_result.json_data is not None:
                for obj in canvas_result.json_data['objects']:
                    if obj['type'] == 'rect':
                        x1 = obj['left'] / st.session_state.params['zoom_factor']
                        y1 = obj['top'] / st.session_state.params['zoom_factor']
                        x2 = (obj['left'] + obj['width']) / st.session_state.params['zoom_factor']
                        y2 = (obj['top'] + obj['height']) / st.session_state.params['zoom_factor']
                        bounding_boxes.append([x1, y1, x2, y2])

                # Limit the number of bounding boxes
                if len(bounding_boxes) > 10:
                    st.warning("Maximum 10 bounding boxes allowed.")
                    bounding_boxes = bounding_boxes[:10]
            else:
                st.info("No bounding boxes added yet.")

            # Button to process the bounding boxes with SAM
            if st.button("Process with SAM"):
                if bounding_boxes:
                    if predictor is None:
                        st.error("SAM model not loaded.")
                    else:
                        st.info("Processing bounding boxes with SAM...")
                        with st.spinner("Processing with SAM..."):
                            try:
                                # Convert bounding boxes to NumPy array
                                bboxes = np.array(bounding_boxes)
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

                                st.success("Masks generated successfully!")

                                # Save masks and colors
                                st.info("Saving masks and colors...")
                                masks_np = np.array(all_masks)
                                mask_colors = np.random.randint(100, 255, (len(masks_np), 3))

                                image_path_obj = Path(st.session_state.image_path)
                                masks_path = image_path_obj.parent / f"{image_path_obj.stem}_masks.npy"
                                colors_path = image_path_obj.parent / f"{image_path_obj.stem}_colors.npy"

                                np.save(masks_path, masks_np)
                                np.save(colors_path, mask_colors)

                                st.info("Projecting masks onto the 3D mesh...")
                                # Project masks to mesh
                                colored_mesh_path = project_masks_to_mesh(
                                    obj_path=st.session_state.obj_path,
                                    masks_path=str(masks_path),
                                    colors_path=str(colors_path),
                                    params_path=st.session_state.camera_params_path,
                                    depth_path=st.session_state.depth_path
                                )

                                st.info("Extracting and saving masked areas...")
                                # Extract and save masked areas
                                output_dir = Path('datasets') / Path(st.session_state.obj_path).stem
                                masks_folders_list = extract_and_save_masked_areas(colored_mesh_path, output_dir, "mask")

                                # Store variables in session state
                                st.session_state.sam_processed = True
                                st.session_state.colored_mesh_path = colored_mesh_path
                                st.session_state.masks_folders_list = masks_folders_list

                                # Clean up memory
                                del all_masks, masks_np, mask_colors
                                gc.collect()

                                st.success("Processing completed successfully!")

                            except Exception as e:
                                st.error(f"Processing error: {str(e)}")
                else:
                    st.warning("Please draw at least one bounding box.")

            # Display Download Buttons if SAM Processing is Done
            if st.session_state.sam_processed:
                st.markdown("### Download Meshes")
                colored_mesh_path = st.session_state.colored_mesh_path
                masks_folders_list = st.session_state.masks_folders_list

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
                    st.info("Rendering 3D visualizations...")
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
                    st.success("3D visualization rendered successfully!")

        except Exception as e:
            st.error(f"Error loading image: {e}")
else:
    st.info("Please upload an `.obj` file to begin.")
