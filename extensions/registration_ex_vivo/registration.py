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
            "reference_images_for_registration_slice_" + str(slice_idx).zfill(2) + ".png",
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
        os.path.join(settings["debug_folder"], "registration_masks_slice_" + str(slice_idx).zfill(2) + ".png"),
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
        for slice_idx in data["slice_integer"].unique():
            reg_file_reference = os.path.join(
                self.settings["session"], "image_registration_reference_slice_" + str(slice_idx).zfill(2) + ".npz"
            )
            # check if the reference images have been saved already
            if os.path.exists(reg_file_reference):
                self.logger.info("Saved registration images found for slice " + str(slice_idx).zfill(2))
                save_path = reg_file_reference
                npzfile = np.load(save_path, allow_pickle=True)
                # this is saved as a dictionary where it should be a numpy archive
                try:
                    reg_images[slice_idx] = npzfile["img_post_reg"]
                    ref_images[slice_idx] = {}
                    ref_images[slice_idx]["image"] = npzfile["ref_images"]
                    self._updata_reg_df(list(reg_images[slice_idx]), slice_idx, npzfile["indices"])
                    continue
                except KeyError:
                    self.logger.info("No registered images found for slice " + str(slice_idx).zfill(2))

            self.logger.info("No saved registration image found for slice " + str(slice_idx).zfill(2))

            images = data[data["slice_integer"] == slice_idx]["image"].values
            phase_images = None
            if self.settings["complex_data"]:
                phase_images = data[data["slice_integer"] == slice_idx]["image_phase"].values

            # Get reference image
            ref_images[slice_idx] = {}
            c_table = data[data["slice_integer"] == slice_idx]
            ref_images[slice_idx]["image"] = c_table.loc[
                data["b_value"] == np.sort(c_table["b_value"])[0], "image"
            ].values[0]

            # if self.settings["complex_data"]:
            #     ref_image_phase = data.loc[
            #         data["b_value"] == np.sort(data[data["slice_integer"] == slice_idx]["b_value"])[0], "image_phase"
            #     ].values[0]
            #     ref_image = ref_image * np.exp(1j * ref_image_phase)

            indices = c_table["diff_config"].values
            self.logger.info(f"Rigid registering slice {slice_idx}")

            assert len(images) == len(indices)

            # registered_images = self._register_itk(
            #     ref_image,
            #     images,
            #     phase_images,
            #     indices,
            #     self.code_path / "extensions" / "image_registration_recipes" / "Elastix_rigid.txt",
            # )

            registered_images = self._register_stackreg(ref_images[slice_idx]["image"], images, phase_images, indices)

            if self.settings["debug"]:
                if not self.settings["complex_data"]:
                    # average of stack pre- and post-registration
                    pre_reg_mag_stack_mean = np.mean(np.stack(images, axis=0), axis=0)
                    mag_list, diff_config_idx_list = map(list, zip(*registered_images))
                    post_reg_mag_stack_mean = np.mean(np.array(mag_list), axis=0)
                else:
                    # TODO do this for complex data
                    pass

                plt.imsave(
                    self.debug_folder / f"reg_rigid_image_{slice_idx:06d}.png",  # noqa
                    np.hstack((pre_reg_mag_stack_mean, post_reg_mag_stack_mean)),
                    cmap="Greys_r",
                )

            # Averaging the repetitions
            average_images = []
            phase_images = []
            for index in np.unique(indices):
                if self.settings["complex_data"]:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_image_real = np.mean([img[0] for img in registered_images_index], axis=0)
                    average_image_imag = np.mean([img[1] for img in registered_images_index], axis=0)
                    average_images.append(np.sqrt(np.square(average_image_real) + np.square(average_image_imag)))
                    phase_images.append(np.arctan2(average_image_imag, average_image_real))
                else:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_images.append(np.mean(np.stack(registered_images_index), axis=0))

            self.logger.info(f"BSpline registering slice {slice_idx}")
            registered_images = self._register_itk(
                ref_images[slice_idx]["image"],
                average_images,
                phase_images,
                registration_mask,
                indices,
                self.code_path / "extensions" / "image_registration_recipes" / "Elastix_bspline.txt",
            )

            if self.settings["debug"]:
                if not self.settings["complex_data"]:
                    # averaged image stack pre- and post-registration
                    pre_reg_mag_stack_mean = np.mean(np.stack(average_images, axis=0), axis=0)
                    mag_list, diff_config_idx_list = map(list, zip(*registered_images))
                    post_reg_mag_stack_mean = np.mean(np.array(mag_list), axis=0)
                else:
                    # TODO do this for complex data
                    pass

                plt.imsave(
                    self.debug_folder / f"reg_elastix_image_{slice_idx:06d}.png",  # noqa
                    np.hstack((pre_reg_mag_stack_mean, post_reg_mag_stack_mean)),
                    cmap="Greys_r",
                )

            # TODO check for complex data
            reg_images[slice_idx] = (
                average_images * np.exp(1j * phase_images) if self.settings["complex_data"] else mag_list
            )

            np.savez(
                reg_file_reference,
                img_post_reg=reg_images[slice_idx],
                ref_images=ref_images[slice_idx]["image"],
                indices=np.unique(indices),
            )
            self.logger.info(f"Saved registered images for slice {slice_idx}")

            self._updata_reg_df(reg_images[slice_idx], slice_idx, np.unique(indices))

            if self.settings["debug"]:
                plt.imsave(
                    self.debug_folder / f"reg_image_{slice_idx: 06d}.png",
                    reg_images[slice_idx][0],
                    cmap="Greys_r",  # noqa
                )

            plot_ref_images(
                np.mean(average_images, axis=0), slice_idx, ref_images[slice_idx]["image"], contour, self.settings
            )

        self.data_reg["diffusion_direction"] = self.data_reg["diffusion_direction"].apply(tuple)
        data_grouped = self.data_reg.groupby(["b_value", "diffusion_direction"])
        self.data_reg["diff_config"] = data_grouped.ngroup()

        self.context["data"] = self.data_reg
        self.context["ref_images"] = ref_images

    def _register_itk(self, ref_image, images, phase_images, mask, indices, recipe):
        ref_image = itk.GetImageFromArray(np.array(ref_image, dtype=np.float32))
        mask = itk.GetImageFromArray(mask)

        # Denoise the images ?
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(recipe.as_posix())

        def register(i):
            mov_image = images[i]

            if self.settings["complex_data"]:
                # TODO we need to fix this for complex data, imag and real
                # need to use the same registration transform
                mov_image_real = mov_image * np.cos(phase_images[i])
                mov_image_imag = mov_image * np.sin(phase_images[i])

                img_reg_real, _ = itk.elastix_registration_method(
                    ref_image.real,
                    itk.GetImageFromArray(mov_image_real),
                    parameter_object=parameter_object,
                    log_to_console=False,
                    fixed_mask=mask,
                )

                img_reg_imag, _ = itk.elastix_registration_method(
                    ref_image.imag,
                    itk.GetImageFromArray(mov_image_imag),
                    parameter_object=parameter_object,
                    log_to_console=False,
                    fixed_mask=mask,
                )
                return img_reg_real, img_reg_imag, indices[i]
            else:
                img_reg, _ = itk.elastix_registration_method(
                    ref_image,
                    itk.GetImageFromArray(np.array(mov_image, dtype=np.float32)),
                    parameter_object=parameter_object,
                    log_to_console=False,
                    fixed_mask=mask,
                )
                img_reg = itk.GetArrayFromImage(img_reg)
                img_reg[img_reg < 0] = 0
                return img_reg, indices[i]

        # registered_images = Parallel(n_jobs=-1)(
        #     delayed(register)(i) for i in tqdm(range(1, len(images)), desc="Registering images")
        # )
        registered_images = [register(i) for i in tqdm(range(0, len(images)), desc="Registering images")]
        return registered_images

    def _register_stackreg(self, ref_image, images, phase_images, indices):
        sr = StackReg(StackReg.TRANSLATION)
        registered_images = []

        for i in tqdm(range(0, len(images)), desc="Registering images"):
            mov_image = images[i]

            if self.settings["complex_data"]:
                # TODO test complex reg
                # get the transform using the magnitude
                sr.register(ref_image, mov_image)

                # apply the above transformation to the real and imag
                mov_image_real = mov_image * np.cos(phase_images[i])
                mov_image_imag = mov_image * np.sin(phase_images[i])

                img_reg_real = sr.transform(mov_image_real)
                img_reg_imag = sr.transform(mov_image_imag)
                registered_images.append(((img_reg_real, img_reg_imag), indices[i]))
            else:
                img_reg = sr.register_transform(ref_image, mov_image)
                registered_images.append((img_reg, indices[i]))

        return registered_images

    def _updata_reg_df(self, reg_images, slice, indices):
        data = self.context["data"]
        # The number of images changed from the input so we need a new dataframe
        b_values = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)]["b_value"].values[0]
            for index in np.unique(indices).astype(int)
        ]
        diffusion_directions = [
            data[(data["diff_config"] == index) & (data["slice_integer"] == slice)]["diffusion_direction"].values[0]
            for index in np.unique(indices).astype(int)
        ]
        if self.settings["complex_data"]:
            self.data_reg = pd.concat(
                [
                    self.data_reg,
                    pd.DataFrame(
                        {
                            "diff_config": np.unique(indices).astype(int),
                            "image": np.abs(reg_images),
                            "image_phase": np.angle(reg_images),
                            "slice_integer": [int(slice) for _ in range(len(reg_images))],
                            "b_value": b_values,
                            "diffusion_direction": diffusion_directions,
                        }
                    ),
                ]
            )
        else:
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
                        }
                    ),
                ]
            )
