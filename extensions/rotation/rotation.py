import os
from numbers import Number
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from extensions.extension_base import ExtensionBase


def rotate_vector(vector: NDArray, angle: Number, axis: str) -> NDArray:
    angle = np.radians(angle)
    if axis == "z":
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    elif axis == "y":
        matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "x":
        matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return np.dot(matrix, vector)


def plot_rotation(image: NDArray, image_rotated: NDArray, index, angle: int, axis: str, settings: Dict):
    nx, ny, nz = image.shape

    fig, ax = plt.subplots(3, 2)
    ax[0][0].imshow(image[nx // 2, :, :], cmap="gray")
    ax[0][0].axis("off")
    ax[1][0].imshow(image[:, ny // 2, :], cmap="gray")
    ax[1][0].axis("off")
    ax[2][0].imshow(image[:, :, nz // 2], cmap="gray")
    ax[2][0].axis("off")

    nx, ny, nz = image_rotated.shape
    ax[0][1].imshow(image_rotated[nx // 2, :, :], cmap="gray")
    ax[0][1].axis("off")
    ax[1][1].imshow(image_rotated[:, ny // 2, :], cmap="gray")
    ax[1][1].axis("off")
    ax[2][1].imshow(image_rotated[:, :, nz // 2], cmap="gray")
    ax[2][1].axis("off")

    ax[0][0].set_title("Original image")
    ax[0][1].set_title("Rotated image")

    fig.suptitle(f"Rotated image {index} by {angle} degrees around the {axis} axis")
    fig.savefig(os.path.join(settings["debug_folder"], f"rotation_{index}_{angle}_{axis}.png"))
    plt.show()


class Rotation(ExtensionBase):
    def run(self) -> None:
        data = self.context["data"]
        # get the rotation angle
        angle = self.settings["rotation_angle"]
        if angle == 90:
            k = 1
        elif angle == 180:
            k = 2
        elif angle == -90:
            k = -1
        else:
            raise ValueError(f"Angle {angle} is not supported.")
        # get the axis
        axis = self.settings["rotation_axis"]

        data_rotated = pd.DataFrame(
            dict(image=[], slice_integer=[], b_value=[], diffusion_direction=[], image_position=[])
        )
        average_images = []
        slices = self.context["slices"]
        print(slices)
        print(data)
        # Build the images
        for index in tqdm(data["diff_config"].unique(), desc="Rotating images"):
            img = [
                np.asarray(data[(data["slice_integer"] == i) & (data["diff_config"] == index)]["image"])[0]
                for i in slices
            ]
            image = np.stack(img, axis=0)
            average_images.append(image)

            # rotate the image
            if axis == "z":
                image_rotated = np.rot90(image, k=k, axes=(0, 1))
            elif axis == "y":
                image_rotated = np.rot90(image, k=k, axes=(0, 2))
            elif axis == "x":
                image_rotated = np.rot90(image, k=k, axes=(1, 2))

            diffusion_direction = rotate_vector(
                data[data["diff_config"] == index]["diffusion_direction"].values[0], angle, axis
            )
            # this is wrong, we need to get the image spacing on the new plane
            # only true if the resolution is isotropic
            image_position = [
                (0.0, 0.0, i * self.context["info"]["slice_thickness"]) for i in range(image_rotated.shape[0])
            ]
            b_value = data[data["diff_config"] == index]["b_value"].values[0]

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
                        )
                    ),
                ]
            )

            # plot the rotation
            if self.settings["debug"]:
                plot_rotation(image, image_rotated, index, angle, axis, self.settings)

        self.context["info"]["img_size"] = image_rotated.shape[1:]
        self.context["info"]["n_slices"] = image_rotated.shape[0]

        self.logger.info(f"Rotated the image by {angle} degrees around the {axis} axis.")

        self.context["data"] = data_rotated
        self.context["slices"] = np.arange(image_rotated.shape[0], dtype=int)
