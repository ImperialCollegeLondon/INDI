import os
import pathlib
from typing import Dict

import itk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pystackreg import StackReg
from tqdm import tqdm

from extensions.extension_base import ExtensionBase
from extensions.extensions import get_snr_maps
from extensions.image_registration import get_registration_mask


def plot_ref_images(image_mean, slice_idx: int, ref_image: NDArray, contour, settings: Dict):
    """

    Parameters
    ----------
    data: dataframe with diffusion info
    ref_images dictionary with all the info on the reference images used
    contour: registration mask contours
    slices array with strings of slice positions
    settings

    Returns
    -------

    """
    # plot reference images
    plt.figure(figsize=(5, 5))
    plt.imshow(ref_image, cmap="Greys_r")
    plt.axis("off")
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "reg_reference_image_slice_" + str(slice_idx).zfill(2) + ".png",
        ),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
    )
    plt.close()

    # plot registration mask

    plt.figure(figsize=(5, 5))
    plt.imshow(image_mean, cmap="Greys_r")
    plt.plot(contour[:, 0], contour[:, 1], "r")
    plt.axis("off")
    plt.savefig(
        os.path.join(settings["debug_folder"], "reg_mask_slice_" + str(slice_idx).zfill(2) + ".png"),
        dpi=200,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close()


class RegistrationExVivo(ExtensionBase):
    def __init__(self, context, settings, logger):
        ExtensionBase.__init__(self, context, settings, logger)

        self.code_path = pathlib.Path(self.settings["code_path"])
        self.debug_folder = pathlib.Path(self.settings["debug_folder"])

    def run(self):
        registration_mask, contour = get_registration_mask(self.context["info"], self.settings)
        self.context["reg_mask"] = registration_mask
        self.context["contour"] = contour

        data = self.context["data"]

        if self.settings["complex_data"]:
            self.data_reg = pd.DataFrame(
                {
                    "diff_config": [],
                    "image": [],
                    "image_phase": [],
                    "slice_integer": [],
                    "b_value": [],
                    "diffusion_direction": [],
                }
            )
        else:
            self.data_reg = pd.DataFrame(
                {
                    "diff_config": [],
                    "image": [],
                    "slice_integer": [],
                    "b_value": [],
                    "diffusion_direction": [],
                }
            )

        reg_images = {}
        ref_images = {}

        self.reg_rigid_df = pd.DataFrame(
            {"image": [], "slice_integer": [], "diff_config": [], "diffusion_direction": [], "b_value_original": []}
        )

        # loop over all slices
        for slice_idx in tqdm(data["slice_integer"].unique(), desc="Registering slices", position=0, leave=True):
            # table with all the data for the current slice
            c_table = data[data["slice_integer"] == slice_idx]

            # get the images and phase images for the current slice
            images = c_table["image"].values
            phase_images = None
            if self.settings["complex_data"]:
                phase_images = data[data["slice_integer"] == slice_idx]["image_phase"].values

            # Get reference image
            ref_images[slice_idx] = {}
            ref_images[slice_idx]["image"] = c_table.loc[
                data["b_value"] == np.sort(c_table["b_value"])[0], "image"
            ].values[0]

            # indices of the different diffusion configurations
            indices = c_table["diff_config"].values
            # index for the lowest b-value
            lower_b_value_index = c_table.loc[data["b_value"] == np.sort(c_table["b_value"])[0], "diff_config"].values[
                0
            ]

            # self.logger.info(f"Rigid registering slice {slice_idx}")

            # check if the registration info has been saved already
            reg_file_path = os.path.join(
                self.settings["session"], "image_registration_slice_" + str(slice_idx).zfill(2) + ".npz"
            )
            if os.path.exists(reg_file_path):
                # self.logger.info("Saved registration images found for slice " + str(slice_idx).zfill(2))
                npzfile = np.load(reg_file_path, allow_pickle=True)
                # this is saved as a dictionary where it should be a numpy archive
                try:
                    reg_images[slice_idx] = npzfile["reg_images"]
                    ref_images[slice_idx] = {}
                    ref_images[slice_idx]["image"] = npzfile["ref_images"]
                    self._update_reg_df(
                        [np.abs(reg_images[slice_idx][i]) for i in range(len(reg_images[slice_idx]))],
                        slice_idx,
                        np.unique(indices),
                    )

                    rigid_reg_images = [
                        npzfile["rigid_reg_images"][i] for i in range(len(npzfile["rigid_reg_images"]))
                    ]
                    self._update_reg_rigid_df(rigid_reg_images, slice_idx, indices, lower_b_value_index)

                    continue
                except KeyError:
                    # self.logger.info("No saved registration images found for slice " + str(slice_idx).zfill(2))
                    pass
            # self.logger.info("No saved registration image found for slice " + str(slice_idx).zfill(2))

            assert len(images) == len(indices)

            # =========================================
            # rigid registration
            # =========================================
            registered_images = self._register_stackreg(ref_images[slice_idx]["image"], images, phase_images, indices)

            # debug mean images pre and post rigid registration
            if self.settings["debug"]:
                if not self.settings["complex_data"]:
                    # average of stack pre- and post-registration
                    pre_reg_mag_stack_mean = np.mean(np.stack(images, axis=0), axis=0)
                    mag_list, diff_config_idx_list = map(list, zip(*registered_images))
                    post_reg_mag_stack_mean = np.mean(np.array(mag_list), axis=0)
                else:
                    # average of stack pre- and post-registration
                    mag = np.stack(images, axis=0)
                    phase = np.stack(phase_images, axis=0)
                    real_mean = np.mean(mag * np.cos(phase), axis=0)
                    imag_mean = np.mean(mag * np.sin(phase), axis=0)
                    pre_reg_mag_stack_mean = np.sqrt(np.square(real_mean) + np.square(imag_mean))

                    real_list, imag_list, diff_config_idx_list = map(list, zip(*registered_images))
                    real_mean_post = np.mean(np.stack(real_list), axis=0)
                    imag_mean_post = np.mean(np.stack(imag_list), axis=0)
                    post_reg_mag_stack_mean = np.sqrt(np.square(real_mean_post) + np.square(imag_mean_post))

                plt.imsave(
                    self.debug_folder / f"reg_rigid_image_{slice_idx:06d}.png",  # noqa
                    np.hstack((pre_reg_mag_stack_mean, post_reg_mag_stack_mean)).repeat(5, axis=0).repeat(5, axis=1),
                    cmap="Greys_r",
                )

            # Averaging the repetitions and storing the rigi registered images for SNR calculation
            average_images = []
            rigid_reg_images = []
            post_averaging_indices = []
            for index in np.unique(indices):
                post_averaging_indices.append(index)
                if self.settings["complex_data"]:
                    registered_images_index = [
                        (img_real, img_imag) for img_real, img_imag, idx in registered_images if idx == index
                    ]
                    list_real = [img[0] for img in registered_images_index]
                    list_imag = [img[1] for img in registered_images_index]
                    average_image_real = np.mean(list_real, axis=0)
                    average_image_imag = np.mean(list_imag, axis=0)
                    mag = np.sqrt(np.square(np.stack(list_real)) + np.square(np.stack(list_imag)))
                    if index == lower_b_value_index:
                        rigid_reg_images += [mag[i] for i in range(len(list_real))]
                    average_images.append(np.sqrt(np.square(average_image_real) + np.square(average_image_imag)))
                else:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_images.append(np.mean(np.stack(registered_images_index), axis=0))
                    if index == lower_b_value_index:
                        rigid_reg_images += registered_images_index

            # save the rigid registered images for later use in calculating SNR
            self._update_reg_rigid_df(rigid_reg_images, slice_idx, indices, lower_b_value_index)

            # self.logger.info(f"BSpline registering slice {slice_idx}")

            # =========================================
            # non-rigid registration
            # =========================================
            registered_images = self._register_itk(
                ref_images[slice_idx]["image"],
                average_images,
                registration_mask,
                self.code_path / "extensions" / "image_registration_recipes" / "Elastix_bspline.txt",
            )

            # debug mean images pre and post non-rigid registration
            if self.settings["debug"]:
                # average of stack pre- and post-registration
                pre_reg_mag_stack_mean = np.copy(post_reg_mag_stack_mean)
                post_reg_mag_stack_mean = np.mean(np.stack(registered_images, axis=0), axis=0)

                plt.imsave(
                    self.debug_folder / f"reg_elastix_image_{slice_idx:06d}.png",  # noqa
                    np.hstack((pre_reg_mag_stack_mean, post_reg_mag_stack_mean)).repeat(5, axis=0).repeat(5, axis=1),
                    cmap="Greys_r",
                )

            # get the arrays
            reg_images[slice_idx] = np.array(registered_images)

            np.savez(
                reg_file_path,
                reg_images=reg_images[slice_idx],
                ref_images=ref_images[slice_idx]["image"],
                rigid_reg_images=rigid_reg_images,
            )
            # self.logger.info(f"Saved registered images for slice {slice_idx}")

            self._update_reg_df(
                [reg_images[slice_idx][i] for i in range(len(reg_images[slice_idx]))],
                slice_idx,
                np.unique(indices),
            )

            if self.settings["debug"]:
                plot_ref_images(
                    np.mean(average_images, axis=0), slice_idx, ref_images[slice_idx]["image"], contour, self.settings
                )

        self.logger.info("Registration Completed")
        self.logger.info("Calculating SNR maps")
        slices = data["slice_integer"].unique()

        # Calculate SNR maps
        mask_3c = np.ones(
            (len(slices), self.context["info"]["img_size"][0], self.context["info"]["img_size"][1]), dtype="uint8"
        )
        self.context["snr"], self.context["noise"], self.context["snr_b0_lv"], self.context["info"] = get_snr_maps(
            self.reg_rigid_df, mask_3c, None, slices, self.settings, self.logger, self.context["info"]
        )

        self.data_reg["diffusion_direction"] = self.data_reg["diffusion_direction"].apply(tuple)

        data_grouped = self.data_reg.groupby(["b_value", "diffusion_direction"])
        self.data_reg["diff_config"] = data_grouped.ngroup()

        self.context["data"] = self.data_reg
        self.context["ref_images"] = ref_images
        self.settings["complex_data"] = False

    def _register_itk(self, ref_image, images, mask, recipe):
        ref_image = itk.GetImageFromArray(np.array(ref_image, order="F", dtype=np.float32))
        mask = itk.GetImageFromArray(np.array(mask, order="F", dtype=np.uint8))

        # TODO do we need the Fortran order here?

        # Denoise the images ?
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(recipe.as_posix())

        def register(i):
            mov_image = images[i]

            # Array must be in Fortran order
            img_reg, _ = itk.elastix_registration_method(
                ref_image,
                itk.GetImageFromArray(np.array(mov_image, order="F", dtype=np.float32)),
                parameter_object=parameter_object,
                log_to_console=False,
                fixed_mask=mask,
            )

            # For some reason the registration is flipped (ITK uses Fortan order)
            img_reg = itk.GetArrayFromImage(img_reg)

            return img_reg.T

        registered_images = [
            register(i) for i in tqdm(range(0, len(images)), desc="Non-rigid reg images", position=2, leave=False)
        ]
        return registered_images

    def _register_stackreg(self, ref_image, images, phase_images, indices):
        sr = StackReg(StackReg.TRANSLATION)
        registered_images = []

        for i in tqdm(range(0, len(images)), desc="Rigid reg images", position=1, leave=False):
            mov_image = images[i]

            if self.settings["complex_data"]:
                # get the transform using the magnitude
                sr.register(ref_image, mov_image)

                # apply the above transformation to the real and imag
                mov_image_real = mov_image * np.cos(phase_images[i])
                mov_image_imag = mov_image * np.sin(phase_images[i])

                img_reg_real = sr.transform(mov_image_real)
                img_reg_imag = sr.transform(mov_image_imag)
                registered_images.append((img_reg_real, img_reg_imag, indices[i]))
            else:
                img_reg = sr.register_transform(ref_image, mov_image)
                registered_images.append((img_reg, indices[i]))

        return registered_images

    def _update_reg_df(self, reg_images, slice, indices):
        data = self.context["data"]

        # The number of images changed from the input so we need a new dataframe
        b_values = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)]["b_value"].values[0]
            for index in np.unique(indices).astype(int)
        ]

        b_values_original = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)]["b_value_original"].values[0]
            for index in np.unique(indices).astype(int)
        ]

        diffusion_directions = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)]["diffusion_direction"].values[0]
            for index in np.unique(indices).astype(int)
        ]

        diffusion_directions_original = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)][
                "diffusion_direction_original"
            ].values[0]
            for index in np.unique(indices).astype(int)
        ]

        self.data_reg = pd.concat(
            [
                self.data_reg,
                pd.DataFrame(
                    {
                        "diff_config": indices.astype(int),
                        "image": reg_images,
                        "slice_integer": [int(slice) for _ in range(len(reg_images))],
                        "b_value": b_values,
                        "diffusion_direction": diffusion_directions,
                        "b_value_original": b_values_original,
                        "diffusion_direction_original": diffusion_directions_original,
                    }
                ),
            ],
            ignore_index=True,
        )

    def _update_reg_rigid_df(self, registered_images, slice_idx, indices, lower_b_value_index):
        data = self.context["data"]

        assert len(registered_images) == len(indices[indices == lower_b_value_index])
        b_values_original = [
            data[(data["diff_config"] == lower_b_value_index) & (data["slice_integer"] == slice_idx)][
                "b_value_original"
            ].values[0]
            for index in indices[indices == lower_b_value_index]
        ]

        diffusion_directions = [
            data[(data["diff_config"] == lower_b_value_index) & (data["slice_integer"] == slice_idx)][
                "diffusion_direction"
            ].values[0]
            for index in indices[indices == lower_b_value_index]
        ]

        assert len(b_values_original) == len(indices[indices == lower_b_value_index])
        assert len(diffusion_directions) == len(indices[indices == lower_b_value_index])
        assert len(b_values_original) == len(indices[indices == lower_b_value_index])

        self.reg_rigid_df = pd.concat(
            [
                self.reg_rigid_df,
                pd.DataFrame(
                    dict(
                        image=registered_images,
                        slice_integer=[slice_idx] * len(registered_images),
                        diff_config=indices[indices == lower_b_value_index],
                        diffusion_direction=diffusion_directions,
                        b_value_original=b_values_original,
                    )
                ),
            ],
            ignore_index=True,
        )
