import numpy as np
import pandas as pd


def complex_averaging(data):
    data[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(data["direction"].tolist(), index=data.index)
    unique_configs = data[["dir_x", "dir_y", "dir_z", "b_value", "slice_integer"]].drop_duplicates()

    data_complex_averaged = []
    for idx, config in unique_configs.iterrows():
        # Select the data for the current configuration
        c_table = data[
            (data["b_value"] == config["b_value"])
            & (data["dir_x"] == config["dir_x"])
            & (data["dir_y"] == config["dir_y"])
            & (data["dir_z"] == config["dir_z"])
            & (data["slice_integer"] == config["slice_integer"])
        ]

        # new column with real and imaginary part
        c_table["image_real"] = c_table["image_phase"].apply(lambda x: np.cos(x))
        c_table["image_real"] = c_table["image_real"] * c_table["image"]
        c_table["image_imag"] = c_table["image_phase"].apply(lambda x: np.sin(x))
        c_table["image_imag"] = c_table["image_imag"] * c_table["image"]

        # average the real and imaginary part
        mean_real = c_table["image_real"].mean()
        mean_imag = c_table["image_imag"].mean()
        mean_img = np.sqrt(np.square(mean_real) + np.square(mean_imag))

        data_complex_averaged.append(
            [
                mean_img,
                # c_table["image"].mean(),
                c_table["b_value"].iloc[0],
                c_table["direction"].iloc[0],
                c_table["image_position"].iloc[0],
                c_table["image_position_label"].iloc[0],
                c_table["slice_integer"].iloc[0],
            ]
        )

    data_complex_averaged = pd.DataFrame(
        data_complex_averaged,
        columns=[
            "image",
            "b_value",
            "direction",
            "image_position",
            "image_position_label",
            "slice_integer",
        ],
    )

    data_complex_averaged["to_be_removed"] = False

    return data_complex_averaged
