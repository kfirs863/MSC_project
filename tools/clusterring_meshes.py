import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import trimesh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from PIL import Image


# Load a pre-trained ResNet model for image feature extraction
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Remove the final classification layer to get feature vectors
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define the image transformation (resize, normalize to match ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),          # Convert the image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


def load_meshes_and_images_from_folders(datasets_folder):
    """Load meshes and corresponding images from a folder hierarchy, and track subfolder names."""
    meshes = []
    depth_images = []
    contour_images_list = []
    skeleton_images = []
    subfolder_names = []

    # Traverse each subfolder in the datasets folder
    for subfolder_name in os.listdir(datasets_folder):
        subfolder_path = os.path.join(datasets_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            ply_file = None
            depth_image_file = None
            contour_image_files = []
            skeleton_image_file = None

            # Traverse files within each subfolder to find mask.ply, depth_profile.png, contour_*.png, and skeleton.png
            for file_name in os.listdir(subfolder_path):
                if file_name == 'mask.ply':  # Mesh file name is 'mask.ply'
                    ply_file = os.path.join(subfolder_path, file_name)
                elif file_name == 'depth_profile.png':  # Depth profile image
                    depth_image_file = os.path.join(subfolder_path, file_name)
                elif file_name.startswith('contour_') and file_name.endswith('.png'):  # Contour images
                    contour_image_files.append(os.path.join(subfolder_path, file_name))
                elif file_name == 'skeleton.png':  # Skeleton image
                    skeleton_image_file = os.path.join(subfolder_path, file_name)

            # Ensure all necessary files are found
            if ply_file and depth_image_file and contour_image_files and skeleton_image_file:
                # Load the mesh using Trimesh instead of Open3D
                mesh = trimesh.load(ply_file)
                meshes.append(mesh)

                # Load the depth profile image
                depth_image = cv2.imread(depth_image_file, cv2.IMREAD_GRAYSCALE)
                depth_images.append(depth_image)

                # Load all contour images
                contour_images = [cv2.imread(contour_file, cv2.IMREAD_GRAYSCALE) for contour_file in contour_image_files]
                contour_images_list.append(contour_images)

                # Load the skeleton image
                skeleton_image = cv2.imread(skeleton_image_file, cv2.IMREAD_GRAYSCALE)
                skeleton_images.append(skeleton_image)

                # Track the subfolder name for tracing back later
                subfolder_names.append(subfolder_name)
            else:
                print(f"Missing files in {subfolder_name}: Mask: {ply_file}, Depth: {depth_image_file}, Contours: {len(contour_image_files)} found, Skeleton: {skeleton_image_file}")

    return meshes, depth_images, contour_images_list, skeleton_images, subfolder_names

def normalize_mesh(mesh):
    """Normalize the mesh to make it scale-invariant."""
    # Scale the mesh to fit within a unit bounding box (1x1x1)
    mesh.apply_scale(1 / mesh.scale)
    return mesh

def compute_approximate_curvature(mesh):
    """Compute approximate curvature for a surface mesh using vertex normals and their neighbors."""
    # Ensure vertex normals are computed
    mesh.rezero()
    mesh.fix_normals()

    # Get vertex neighbors (adjacent vertices)
    adjacency = mesh.vertex_adjacency_graph

    # Compute curvature as the difference between vertex normals of adjacent vertices
    normals = mesh.vertex_normals
    curvature = []

    for vertex_idx in range(len(normals)):
        neighbors = adjacency[vertex_idx]
        normal_variations = [np.linalg.norm(normals[vertex_idx] - normals[neighbor]) for neighbor in neighbors]
        curvature.append(np.mean(normal_variations))

    return np.array(curvature)

def extract_geometric_features(meshes):
    """Extract geometric features like volume, surface area, PCA components, curvature, etc."""
    features = []

    for mesh in meshes:
        # Normalize the mesh to remove size differences
        mesh = normalize_mesh(mesh)

        # Surface area of the mesh
        surface_area = mesh.area

        # PCA on vertices to capture shape information
        verts = mesh.vertices
        pca = PCA(n_components=min(verts.shape[0], 3))  # Get the top 3 PCA components
        pca.fit(verts)
        pca_components = pca.explained_variance_ratio_

        # Combine all features into a single vector
        feature_vector = np.concatenate([[surface_area], pca_components])

        features.append(feature_vector)

    return np.vstack(features)


def extract_image_features(images):
    """Extract features from images (ndarray) using a pre-trained ResNet model."""
    image_features = []

    for image in images:
        # Ensure the image is a valid ndarray
        if isinstance(image, np.ndarray):
            # OpenCV loads images as BGR, but ResNet expects RGB, so we need to convert the color channels
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image (ndarray) to a PIL Image
            pil_image = Image.fromarray(image)

            # Apply transformations and extract features using the pre-trained ResNet model
            input_tensor = transform(pil_image).unsqueeze(0)  # Add a batch dimension
            with torch.no_grad():  # Disable gradient computation
                features = model(input_tensor).squeeze().numpy()  # Extract ResNet features

            image_features.append(features)
        else:
            print(f"Warning: Image is not a valid ndarray. Skipping.")

    return np.array(image_features)


def extract_combined_contour_features(contour_images_list):
    """Extract and combine features from multiple contour images."""
    combined_contour_features = []

    for contour_images in contour_images_list:
        # Extract features from each contour image using the pre-trained ResNet
        contour_features = extract_image_features(contour_images)
        # Combine the features from multiple contour images (e.g., by averaging or concatenation)
        combined_features = np.mean(contour_features, axis=0)  # Here, we're using averaging
        combined_contour_features.append(combined_features)

    return np.array(combined_contour_features)


def visualize_clustered_meshes(meshes, cluster_labels, subfolder_names):
    """Visualize the clustered meshes in 3D, colored by their cluster labels using Open3D."""
    # Create a unique color for each cluster
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.get_cmap('viridis', num_clusters)

    # List to store the Open3D mesh objects with colors applied
    colored_meshes = []

    for i, mesh in enumerate(meshes):
        # Get RGB color from the colormap based on the cluster label
        color = colors(cluster_labels[i])[:3]  # Get RGB values in [0, 1] range
        color = np.array(color)  # Ensure it is an array

        # Convert the mesh to Open3D mesh (if needed, depending on the original format)
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices),
                                                 triangles=o3d.utility.Vector3iVector(mesh.faces))
        else:
            mesh_o3d = mesh

        # Apply the color to the mesh
        mesh_o3d.paint_uniform_color(color)

        # Append to the list of colored meshes
        colored_meshes.append(mesh_o3d)

    # Visualize the meshes with Open3D
    o3d.visualization.draw_geometries(colored_meshes, window_name="Clustered Meshes", mesh_show_back_face=True)

    # Print out the subfolder names with their corresponding cluster labels
    for tag, label in zip(subfolder_names, cluster_labels):
        print(f"Subfolder: {tag}, Cluster label: {label}")


def main():
    # Directory containing subfolders with datasets
    datasets_folder = "/mobileye/RPT/users/kfirs/kfir_project/MSC_Project/datasets"

    # Load the meshes and corresponding images from subfolders
    meshes, depth_images, contour_images_list, skeleton_images, subfolder_names = load_meshes_and_images_from_folders(datasets_folder)

    # Extract geometric features from meshes
    geometric_features = extract_geometric_features(meshes)

    # Extract image features from depth profile images using ResNet
    depth_features = extract_image_features(depth_images)

    # Extract and combine features from contour images
    contour_features = extract_combined_contour_features(contour_images_list)

    # Extract features from skeleton images
    skeleton_features = extract_image_features(skeleton_images)

    # Combine all the features: geometric, depth profile, contour, and skeleton
    combined_features = np.hstack((geometric_features, depth_features, contour_features, skeleton_features))

    combined_features = depth_features
    # Standardize the combined features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # Apply PCA to reduce the dimensionality of the combined features
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    reduced_features = pca.fit_transform(combined_features)
    print(f"Reduced features shape: {reduced_features.shape}")

    # Perform clustering on the reduced features
    num_clusters = 3  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters,n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Calculate and print clustering evaluation metrics
    silhouette_avg = silhouette_score(reduced_features, cluster_labels)
    db_score = davies_bouldin_score(reduced_features, cluster_labels)
    inertia = kmeans.inertia_

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"KMeans Inertia: {inertia}")

    # Visualize clustered meshes with their subfolder names
    visualize_clustered_meshes(meshes, cluster_labels, subfolder_names)

if __name__ == "__main__":
    main()
