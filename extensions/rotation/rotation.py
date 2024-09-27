import os
from numbers import Number
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from extensions.extension_base import ExtensionBase
from extensions.image_registration import get_registration_mask


def rotate_vector(vector: NDArray, angle: Number, axis: str) -> NDArray:
    # the rotation matrices need to be left hand coordinates
    angle = np.radians(angle)
    if axis == "z":
        rot_matrix = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    elif axis == "y":
        rot_matrix = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    elif axis == "x":
        rot_matrix = np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])
    return np.dot(rot_matrix, vector)


def plot_rotation(images: NDArray, images_rotated: NDArray, angle: int, axis: str, settings: Dict):
    nz_pre, nx_pre, ny_pre = images.shape
    nz_post, nx_post, ny_post = images_rotated.shape

    fig = plt.figure(layout="constrained", figsize=(10, 10))

    # Make two subfigures, left ones more narrow than right ones:
    sfigs = fig.subfigures(1, 2, width_ratios=[1, 1])

    # Add subplots to left subfigure:
    lax = sfigs[0].subplots(3, 1)
    sfigs[0].suptitle("Pre-rotation")

    # Add subplots to right subfigure:
    rax = sfigs[1].subplots(3, 1)
    sfigs[1].suptitle("Post-rotation")

    lax[0].imshow(images[nz_pre // 2, :, :], cmap="gray", aspect="auto")
    lax[0].axis("off")
    lax[1].imshow(images[:, nx_pre // 2, :], cmap="gray", aspect="auto")
    lax[1].axis("off")
    lax[2].imshow(images[:, :, ny_pre // 2], cmap="gray", aspect="auto")
    lax[2].axis("off")

    rax[0].imshow(images_rotated[nz_post // 2, :, :], cmap="gray", aspect="auto")
    rax[0].axis("off")
    rax[1].imshow(images_rotated[:, nx_post // 2, :], cmap="gray", aspect="auto")
    rax[1].axis("off")
    rax[2].imshow(images_rotated[:, :, ny_post // 2], cmap="gray", aspect="auto")
    rax[2].axis("off")

    fig.suptitle(f"Rotation by {angle} degrees around the {axis} axis")

    fig.savefig(os.path.join(settings["debug_folder"], "data_rotation_mid_slice_lowest_b_value.png"), dpi=200)
    plt.close()


class Rotation(ExtensionBase):
    def run(self) -> None:
        if self.settings["rotate"]:
            data = self.context["data"]
            ref_images = self.context["ref_images"]
            ref_images_array = np.stack([ref_images[i]["image"] for i in self.context["slices"]], axis=0)

            # get the rotation angle
            angle = self.settings["rotation_angle"]
            if angle == 90:
                rot_k = 1
            elif angle == 180:
                rot_k = 2
            elif angle == -90:
                rot_k = -1
            else:
                raise ValueError(f"Angle {angle} is not supported.")

            # get the axis
            axis = self.settings["rotation_axis"]

            # initiate table with new rotated data
            data_rotated = pd.DataFrame(
                dict(image=[], slice_integer=[], b_value=[], diffusion_direction=[], image_position=[], diff_config=[])
            )

            # store mean image before and after rotation for the lowes b-value
            average_images_pre_rot = []
            average_images_post_rot = []
            lower_b_value_index = data.loc[data["b_value"] == np.sort(data["b_value"])[0], "diff_config"].values[0]

            slices = self.context["slices"]
            data["slice_integer"] = data["slice_integer"].apply(int)

            self.rotate_snr(axis, rot_k)

            # Build the rotated images
            # loop over each diffusion configuration
            for index in tqdm(data["diff_config"].unique(), desc="Rotating images"):
                # get the volume for the current diffusion configuration
                img = [
                    np.asarray(data[(data["slice_integer"] == i) & (data["diff_config"] == index)]["image"])
                    for i in slices
                ]
                image = np.stack([i[0] for i in img if i.shape[0] != 0], axis=0)

                # rotate the image
                if axis == "z":
                    image_rotated = np.rot90(image, k=rot_k, axes=(0, 1))
                    ref_images_array = np.rot90(ref_images_array, k=rot_k, axes=(0, 1))

                    # get slice spacing
                    image_positions = []
                    for key_, name_ in self.context["info"]["integer_to_image_positions"].items():
                        image_positions.append(name_)
                    # calculate distances between slices
                    spacing_z = [
                        np.sqrt(
                            (image_positions[i][0] - image_positions[i + 1][0]) ** 2
                            + (image_positions[i][1] - image_positions[i + 1][1]) ** 2
                            + (image_positions[i][2] - image_positions[i + 1][2]) ** 2
                        )
                        for i in range(len(image_positions) - 1)
                    ]
                    new_slice_spacing = int(np.mean(spacing_z))

                elif axis == "y":
                    image_rotated = np.rot90(image, k=rot_k, axes=(0, 2))
                    ref_images_array = np.rot90(ref_images_array, k=rot_k, axes=(0, 2))
                    new_slice_spacing = self.context["info"]["pixel_spacing"][0]

                elif axis == "x":
                    image_rotated = np.rot90(image, k=rot_k, axes=(1, 2))
                    ref_images_array = np.rot90(ref_images_array, k=rot_k, axes=(1, 2))
                    new_slice_spacing = self.context["info"]["pixel_spacing"][1]

                diffusion_direction = rotate_vector(
                    data[data["diff_config"] == index]["diffusion_direction"].values[0], angle, axis
                )

                image_position = [(0.0, 0.0, i * new_slice_spacing) for i in range(image_rotated.shape[0])]
                b_value = data[data["diff_config"] == index]["b_value"].values[0]
                b_value_original = data[data["diff_config"] == index]["b_value_original"].values[0]
                diffusion_direction_original = data[data["diff_config"] == index][
                    "diffusion_direction_original"
                ].values[0]

                data_rotated = pd.concat(
                    [
                        data_rotated,
                        pd.DataFrame(
                            dict(
                                image=[image_rotated[i, :, :] for i in range(image_rotated.shape[0])],
                                b_value=[b_value for _ in range(image_rotated.shape[0])],
                                diffusion_direction=[diffusion_direction for _ in range(image_rotated.shape[0])],
                                image_position=image_position,
                                slice_integer=np.arange(image_rotated.shape[0], dtype=int),
                                b_value_original=[b_value_original for _ in range(image_rotated.shape[0])],
                                diffusion_direction_original=[
                                    diffusion_direction_original for _ in range(image_rotated.shape[0])
                                ],
                            )
                        ),
                    ],
                    ignore_index=True,
                )

                if index == lower_b_value_index:
                    average_images_pre_rot.append(image)
                    average_images_post_rot.append(image_rotated)

            # plot the rotation
            if self.settings["debug"]:
                plot_rotation(
                    average_images_pre_rot[0],
                    average_images_post_rot[0],
                    angle,
                    axis,
                    self.settings,
                )

            self.context["info"]["img_size"] = image_rotated.shape[1:]
            self.context["info"]["n_slices"] = image_rotated.shape[0]
            self.context["reg_mask"], _ = get_registration_mask(self.context["info"], self.settings)
            self.context["ref_images"] = {i: {"image": ref_images_array[i]} for i in range(len(ref_images_array))}
            self.logger.info(f"Rotated the image by {angle} degrees around the {axis} axis.")

            data_rotated["diffusion_direction"] = data_rotated["diffusion_direction"].apply(tuple)
            data_grouped = data_rotated.groupby(["b_value", "diffusion_direction"])
            data_rotated["diff_config"] = data_grouped.ngroup()

            self.context["data"] = data_rotated
            self.context["slices"] = np.arange(image_rotated.shape[0], dtype=int)

    def rotate_snr(self, axis, rot_k):
        slices = self.context["slices"]
        dti = self.context["dti"]

        snr = {}

        for i in slices:
            for k in dti["snr"][i]:
                snr[k] = {}

        for i in slices:
            for k in dti["snr"][i]:
                snr[k][i] = dti["snr"][i][k]

        for k in snr:
            s = list(sorted(snr[k].items()))
            snr[k] = [ss[1] for ss in s]
        snr = pd.DataFrame(snr)

        snr_arrays = {}
        for key in snr:
            snr_image = np.stack([snr[key].values[i] for i in range(len(snr[key]))])
            if axis == "z":
                snr_arrays[key] = np.rot90(snr_image, k=rot_k, axes=(0, 1))
            elif axis == "y":
                snr_arrays[key] = np.rot90(snr_image, k=rot_k, axes=(0, 2))
            elif axis == "x":
                snr_arrays[key] = np.rot90(snr_image, k=rot_k, axes=(1, 2))

        snr = {}
        for i in range(len(snr_arrays[key])):
            snr[i] = {k: snr_arrays[k][i, ...] for k in snr_arrays.keys()}

        self.context["dti"]["snr"] = snr
