import logging
from typing import Dict, List, Number

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


def rotate_data(data: pd.DataFrame, slices: List[int], settings: Dict, logger: logging.Logger) -> pd.DataFrame:
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

    # rotate the data
    image = np.asarray([np.asarray(data["image"][i], dtype=int) for i in slices])

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
            data["diffusion_direction"][i] = rotate_vector(data["diffusion_direction"][i], angle, axis)
        logger.info(f"Rotated the vectors by {angle} degrees around the {axis} axis.")

    # update the data
    data["image_rotation"] = image_rotated

    return data, slices
