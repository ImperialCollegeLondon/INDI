import os

import numpy as np
from denoise.nlm_tensor import nlm_tensor

from indi.extensions.tensor_fittings import plot_tensor_components


def tensor_denoising(dti, slices, average_images, mask_3c, logger, settings):

    logger.debug("Tensor denoising with NLM")

    tensor = dti["tensor"]

    np.save(os.path.join(settings["debug_folder"], "tensor.npy"), tensor)

    denoising_settings = {
        "h": 3,
        "patch_size": 5,
        "window_size": 15,
    }
    if "tensor_denoising_settings" in settings:
        denoising_settings.update(settings["tensor_denoising_settings"])
    tensor_denoised = nlm_tensor(tensor, **denoising_settings)

    dti["tensor"] = tensor_denoised

    if settings["debug"]:
        plot_tensor_components(
            tensor_denoised,
            average_images,
            mask_3c,
            slices,
            settings,
            "tensor_denoising_",
        )

    return dti
