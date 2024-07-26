import pyvista as pv


def generate_realistic_orthoimage(mesh_path, texture_path):
    # Load the mesh
    mesh = pv.read(mesh_path)

    # Check if the mesh has texture coordinates
    # if mesh.has_text_coords:
    # Load the texture
    texture = pv.read_texture(texture_path)

    # Setup the plotter for off-screen rendering
    plotter = pv.Plotter(off_screen=True)

    # Add the mesh with the texture
    plotter.add_mesh(mesh)

    # Custom lighting to enhance the textured look
    light = pv.Light(position=(1, 1, 1), intensity=1)
    plotter.add_light(light)

    # Set the camera to view along the XY plane
    plotter.view_xy()

    # Capture the orthoimage as a screenshot and save directly to a file
    plotter.screenshot("orthoimage.png")

    print("Realistic orthoimage has been saved as 'realistic_orthoimage.png'.")
    # else:
    #     print("Mesh does not have texture coordinates, cannot apply texture.")


# Example usage of the function
generate_realistic_orthoimage('/mobileye/RPT/users/kfirs/temp/S01/S01.obj', '/mobileye/RPT/users/kfirs/temp/textures/brick2_brick.jpeg')

