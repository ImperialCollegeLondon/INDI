import os

import numpy as np
import pyvista as pv
from numpy.typing import NDArray


def save_vtk_file(
    vectors: dict, tensors: dict, scalars: dict, info: dict, name: str, folder_path: str, save_mesh: bool = False
):
    """
    Save vector, tensor, and scalar fields as a VTK file for visualization.

    This function creates a PyVista ImageData mesh from the provided scalar, vector, and tensor fields,
    applies coordinate system rotation, and saves the result as a VTK file. Optionally, it can also
    extract and save a smoothed surface mesh as a PLY file.

    Args:
        vectors (dict): Dictionary of vector fields to save (e.g., eigenvectors).
        tensors (dict): Dictionary of tensor fields to save (e.g., diffusion tensors).
        scalars (dict): Dictionary of scalar fields to save (e.g., FA, MD, mask).
        info (dict): Dictionary containing image metadata (e.g., pixel spacing, slice spacing).
        name (str): Base name for the output file.
        folder_path (str): Path to the folder where files will be saved.
        save_mesh (bool, optional): If True, also saves a smoothed surface mesh as a PLY file. Defaults to False.

    Returns:
        None
    """

    # shape of the vector field
    # [slice, row, column, xyz]
    img_shape = scalars[next(iter(scalars))].shape

    # Create a mesh and add the scalars, vectors and tensors
    mesh = pv.ImageData()
    mesh.dimensions = np.array((img_shape[1], img_shape[2], img_shape[0])) + 1
    mesh.origin = (0, 0, 0)  # The bottom left corner of the data set
    mesh.spacing = (info["pixel_spacing"][0], info["pixel_spacing"][1], info["slice_spacing"])

    # add all scalar maps
    for scalar_idx, scalar_name in enumerate(scalars):
        mesh.cell_data[scalar_name] = scalars[scalar_name].transpose(1, 2, 0).flatten(order="F")

    # rotation to match coordinates directions
    # x positive left to right, y positive bottom to top, z positive away from you when looking from the feet
    rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])

    # add all vector fields
    for vector_idx, vector_name in enumerate(vectors):
        c_vector = vectors[vector_name].copy()
        c_vector = c_vector.transpose(1, 2, 0, 3).reshape((img_shape[1] * img_shape[2] * img_shape[0], 3), order="F")

        c_vector = np.matmul(c_vector, rot_mat)
        mesh.cell_data[vector_name] = c_vector

    # add all tensor fields
    for tensor_idx, tensor_name in enumerate(tensors):
        c_tensor = tensors[tensor_name].copy()
        c_tensor = np.reshape(c_tensor, (img_shape[0], img_shape[1], img_shape[2], 3, 3))
        c_tensor = np.matmul(rot_mat, np.matmul(c_tensor, rot_mat.T))
        c_tensor = np.reshape(c_tensor, (img_shape[0], img_shape[1], img_shape[2], 9))
        c_tensor = c_tensor.transpose(1, 2, 0, 3).reshape((img_shape[1] * img_shape[2] * img_shape[0], 9), order="F")
        mesh.cell_data[tensor_name] = c_tensor

    # save mesh as VTK file
    mesh.save(os.path.join(folder_path, name + ".vtk"))

    # save surface mesh as png
    if save_mesh:
        vol = mesh.threshold(value=1, scalars="mask", invert=False)
        surf = vol.extract_geometry()
        surf_smooth = surf.smooth_taubin(n_iter=50, pass_band=0.05)

        surf_smooth.save(os.path.join(folder_path, "surface_mesh" + ".ply"))

    if "primary_evs" in vectors:
        # save primary eigenvector field
        # it will only save the vectors in the surface, so if multislice we won't see all the vectors
        vol = mesh.threshold(value=1, scalars="mask", invert=False)
        surf = vol.extract_geometry()
        surf.set_active_vectors("primary_evs")
        surf.arrows.save(os.path.join(folder_path, "primary_eigenvectors" + ".ply"))

        # unused pyvista plots
        # pl = pv.Plotter(off_screen=True, window_size=(2000,1000))
        # pl.add_mesh(surf_smooth, show_edges=True, show_scalar_bar=False, opacity=0.5, color="w")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['circ'], mag=2, color="r", label="Circumferential EVs")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['long'], mag=2, color="g", label="Longitudinal EVs")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['radi'], mag=2, color="b", label="Radial EVs")
        # pl.add_legend()
        # pl.camera.position = (105.83953800289639, -37.88726518236088, -26.260429526179585)
        # pl.camera.focal_point = (33.501551389694214, 33.497819662094116, 11.954181253910065)
        # pl.camera.view_up = (-0.09265931623571144, 0.39531983694024236, -0.9138580183137153)
        # pl.show(screenshot=os.path.join(folder_path, "surface_mesh" + ".png"))

        # pl = pv.Plotter(window_size=(3000,2000))
        # pl.add_mesh(surf_smooth, show_edges=True, show_scalar_bar=False, opacity=0.5, color="w")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['circ'], mag=2, color="r", label="Circumferential EVs")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['long'], mag=2, color="g", label="Longitudinal EVs")
        # pl.add_arrows(mesh.cell_centers().points, mesh.cell_data['radi'], mag=2, color="b", label="Radial EVs")
        # pl.add_legend()
        # pl.camera.position = (105.83953800289639, -37.88726518236088, -26.260429526179585)
        # pl.camera.focal_point = (33.501551389694214, 33.497819662094116, 11.954181253910065)
        # pl.camera.view_up = (-0.09265931623571144, 0.39531983694024236, -0.9138580183137153)
        # a = pl.show(return_cpos=True)

        # pl = pv.Plotter()
        # pl.add_mesh(surf_smooth, show_edges=True, show_scalar_bar=False, opacity=0.5, color="w")
        # pl.add_arrows(mesh.cell_centers().points, vec, mag=1, color="r", label="Primary EVs")
        # pl.camera.zoom(2.0)
        # pl.show(auto_close=False)
        # path = pl.generate_orbital_path(n_points=36, shift=mesh.length)
        # pl.open_gif(os.path.join(folder_path, name + ".gif"))
        # pl.orbit_on_path(path, write_frames=True)
        # pl.close()
        # # pl.show()


def export_vectors_tensors_vtk(dti: dict, info: dict, settings: dict, mask_3c: NDArray, average_images: NDArray):
    """
    Organize and export DTI maps, vectors, tensors, and scalars in VTK format for visualization.

    This function prepares the primary, secondary, and tertiary eigenvectors, the diffusion tensor,
    and various scalar maps (e.g., HA, TA, FA, MD, mask, S0, etc.) and exports them as a VTK file
    using `save_vtk_file`. The output can be visualized in tools such as ParaView.

    Args:
        dti (dict): Dictionary containing DTI maps and derived quantities (e.g., eigenvectors, tensor, FA, MD, etc.).
        info (dict): Dictionary with image metadata (e.g., pixel spacing, slice spacing).
        settings (dict): Configuration and output settings.
        mask_3c (NDArray): 3-class segmentation mask array.
        average_images (NDArray): Array of average images for each slice.

    Returns:
        None
    """
    vectors = {}
    vectors["primary_evs"] = dti["eigenvectors"][:, :, :, :, 2]
    vectors["secondary_evs"] = dti["eigenvectors"][:, :, :, :, 1]
    vectors["tertiary_evs"] = dti["eigenvectors"][:, :, :, :, 0]

    required_shape = (
        dti["tensor"].shape[0],
        dti["tensor"].shape[1],
        dti["tensor"].shape[2],
        dti["tensor"].shape[3] * dti["tensor"].shape[4],
    )
    tensor_mat = np.reshape(dti["tensor"], required_shape)
    tensors = {"diff_tensor": tensor_mat}

    maps = {}
    maps["HA"] = dti["ha"]
    maps["TA"] = dti["ta"]
    maps["E2A"] = dti["e2a"]
    maps["MD"] = dti["md"]
    maps["FA"] = dti["fa"]
    maps["mask"] = mask_3c
    maps["s0"] = dti["s0"]
    maps["mag_image"] = average_images
    maps["mode"] = dti["mode"]
    maps["frob_norm"] = dti["frob_norm"]
    maps["mag_anisotropy"] = dti["mag_anisotropy"]
    maps["bullseye"] = dti["bullseye"]
    maps["distance_endo"] = dti["distance_endo"]
    maps["distance_epi"] = dti["distance_epi"]
    maps["distance_transmural"] = dti["distance_transmural"]

    save_vtk_file(
        vectors, tensors, maps, info, "eigensystem", os.path.join(settings["results"], "data"), save_mesh=False
    )
