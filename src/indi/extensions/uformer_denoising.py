import numpy as np
from numpy.typing import NDArray

from indi.extensions.extensions import crop_pad_rotate_array, reshape_tensor_from_6_to_3x3
from indi.extensions.uformer_tensor_denoising.uformer_tensor_denoising import main as uformer_main


def denoise_tensor(D: NDArray, settings: dict) -> NDArray:
    """
    Denoise tensor with MTs Uformer models

    Args:
        D: original tensors
        settings: settings for the denoising process

    Returns:
        D_denoised: denoised tensors

    """

    # Make the tensor H & W [128, 128]
    initial_shape = D.shape
    new_shape = (initial_shape[0], 128, 128, initial_shape[3], initial_shape[4])
    D_new = crop_pad_rotate_array(D, new_shape, False)

    # Reorder the dimensions of the tensor (N, C, H, W)
    D_new = np.transpose(D_new, (0, 3, 4, 1, 2))

    # run uformer denoising
    breath_holds = settings["uformer_breatholds"]
    D_denoised = uformer_main(breath_holds, D_new)

    # revert back tensor to the usual dim order
    D_denoised = np.transpose(D_denoised, (0, 2, 3, 1))

    # convert last dim from 6 to 3x3
    D_denoised = reshape_tensor_from_6_to_3x3(D_denoised)

    D_denoised = crop_pad_rotate_array(D_denoised, initial_shape, False)

    return D_denoised
