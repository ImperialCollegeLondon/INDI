from typing import Union

import numpy as np
import torch

from indi.extensions.uformer_tensor_denoising.utils import IndependentEnsemble

# need to have the line below, otherwise I get a python error:
# /opt/homebrew/Cellar/python@3.11/3.11.12/Frameworks/Python.framework/Versions/3.11/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up
torch.set_num_threads(1)

MODELS = {
    (1, "1BH", "BH1"): "/usr/local/dtcmr/transformer_tensor_denoising/state_dict_1bh.pth",
    (3, "3BH", "BH3"): "/usr/local/dtcmr/transformer_tensor_denoising/state_dict_3bh.pth",
    (5, "5BH", "BH5"): "/usr/local/dtcmr/transformer_tensor_denoising/state_dict_5bh.pth",
}

SHIFT = torch.tensor(
    [
        0.08041676878929138,
        -0.0007398675661534071,
        0.00014480308163911104,
        0.08055038005113602,
        0.0005092238425277174,
        0.08177676051855087,
    ]
).view(1, 6, 1, 1)

SCALE = torch.tensor(
    [
        0.21778962016105652,
        0.05770166590809822,
        0.050258882343769073,
        0.2194574773311615,
        0.04925113171339035,
        0.2210494726896286,
    ]
).view(1, 6, 1, 1)


MODEL_CONFIG = {
    "discriminator": {
        "depth": 4,
        "n_channels": 6,
        "image_size": 128,
        "initial_channels": 16,
        "_convert_": "all",
        "_target_": "indi.extensions.uformer_tensor_denoising.discriminator.DiscriminatorForVGG",
    },
    "generator": {
        "img_size": 128,
        "in_chans": 6,
        "out_chans": 6,
        "embed_dim": 16,
        "depths": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "num_heads": [1, 2, 4, 8, 16, 16, 8, 4, 2],
        "win_size": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "patch_norm": True,
        "use_checkpoint": False,
        "token_projection": "linear",
        "token_mlp": "leff",
        "se_layer": False,
        "_convert_": "all",
        "_target_": "indi.extensions.uformer_tensor_denoising.uformer.Uformer",
    },
    "_convert_": "all",
    "_target_": "indi.extensions.uformer_tensor_denoising.gan_wrapper.GANUFormerModelWrapper",
}


def get_checkpoint_path(model_name: Union[str, int]) -> str:
    for keys in MODELS.keys():
        if model_name in keys:
            return MODELS[keys]

    raise ValueError(f"Model name {model_name} not found!")


def load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=device)

    model = IndependentEnsemble(
        MODEL_CONFIG,
        models_n=5,
        isgan=True,
    )

    model.load_state_dict(state_dict)
    model.current_model_idx = model.models_n - 1

    model = model.to(device)
    model.eval()

    return model


def inference(model: torch.nn.Module, data: np.ndarray) -> np.ndarray:
    """
    Args:
        model: model to use for inference
        data: data to use for inference. This should contain a numpy array of shape (N, ..., H, W)
            where N is the number of samples, C is the number of channels, H is the height and W is the width.
            H and W HAVE to be 128x128.
            ... can either be equal to 6 (unique entries in the diffusion matrix), 9 (full diffusion matrix)
            or (3, 3) (full diffusion matrix in a 3x3 matrix).

    Returns:
        output_data: numpy array of shape (N, C, H, W) with the predictions of the model.
            notice that here C is equal to 6 and H and W will be 128x128.
    """
    if data.shape[1] == 9:
        data = data[:, np.array([0, 1, 2, 4, 5, 8])]
    elif data.shape[1] == 3 and data.shape[2] == 3:
        data = data.reshape((-1, 9, 128, 128))
        data = data[:, np.array([0, 1, 2, 4, 5, 8])]
    elif data.shape[1] != 6:
        raise ValueError(
            f"Data shape {data.shape} not supported! Expected shape (N, ..., H, W) with H and W 128x128 and "
            f"... either 6, 9 or (3, 3)."
        )

    assert (
        data.shape[-2] == data.shape[-1] == 128
    ), f"Data shape {data.shape} not supported! Expected shape (N, ..., H, W) with H and W 128x128 and ... either 6, 9 or (3, 3)."

    model.eval()
    device = model.parameters().__next__().device
    output_data = np.zeros_like(data)
    for i, datapoint in enumerate(data):
        datapoint = torch.from_numpy(datapoint).float().to(device).unsqueeze(0)

        datapoint *= 500
        datapoint -= SHIFT.to(device)
        datapoint /= SCALE.to(device)

        prediction = model(datapoint)

        prediction *= SCALE.to(device)
        prediction += SHIFT.to(device)
        prediction /= 500

        prediction = prediction.detach().cpu().numpy()
        output_data[i] = prediction

    return output_data


def main(breath_holds: Union[int, str], data: np.ndarray) -> np.ndarray:
    checkpoint_path = get_checkpoint_path(breath_holds)
    trained_model = load_checkpoint(checkpoint_path, "cpu")
    output_data = inference(trained_model, data)
    return output_data


if __name__ == "__main__":
    breath_holds = 1
    data = np.random.rand(1, 6, 128, 128)
    output_data = main(breath_holds, data)
    print(output_data.shape)
