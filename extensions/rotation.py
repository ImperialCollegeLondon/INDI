import logging
import os
from numbers import Number
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def rotate_vector(vector: NDArray, angle: Number, axis: str):
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


def rotate_data(
    data: pd.DataFrame, slices: List[int], settings: Dict, logger: logging.Logger
) -> Tuple[pd.DataFrame, List[int]]:
    # get the rotation angle
    angle = settings["rotation_angle"]
    if angle == 90:
        k = 1
    elif angle == 180:
        k = 2
    elif angle == -90:
        k = -1
    else:
        raise ValueError(f"Angle {angle} is not supported.")
    # get the axis
    axis = settings["rotation_axis"]

    print(data["index"])
    plt.plot(data["index"])
    plt.show()
    # rotate the data
    for index in data["index"]:
        print(data[(index == data["index"]) & (0 == data["slice_integer"])]["image"])
        image = np.asarray(
            [
                np.asarray(data[(index == data["index"]) & (i == data["slice_integer"])]["image"], dtype=int)
                for i in slices
            ]
        )

        # rotate the image
        if axis == "z":
            image_rotated = np.rot90(image, k=k, axes=(0, 1))
        elif axis == "y":
            image_rotated = np.rot90(image, k=k, axes=(0, 2))
        elif axis == "x":
            image_rotated = np.rot90(image, k=k, axes=(1, 2))

        logger.info(f"Rotated the image by {angle} degrees around the {axis} axis.")

        # rotate the vectors
        if "diffusion_direction" in data.columns:
            for i in slices:
                data[(index == data["index"]) & (i == data["slice_integer"])]["diffusion_direction"] = rotate_vector(
                    data["diffusion_direction"][i], angle, axis
                )
            logger.info(f"Rotated the vectors by {angle} degrees around the {axis} axis.")

        # plot the rotation
        if settings["debug"]:
            plot_rotation(image, image_rotated, index, angle, axis, settings)

    # Re build the data frame for the rotated image

    # update the data
    data["image_rotation"] = image_rotated

    return data, slices
