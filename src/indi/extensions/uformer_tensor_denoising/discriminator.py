from functools import partial
from math import prod

import torch
import torch.nn as nn


class DiscriminatorForVGG(nn.Module):
    def __init__(self, depth, n_channels, image_size, complex_data=False, initial_channels=32, **kwargs):
        super(DiscriminatorForVGG, self).__init__()

        self.n_channels = n_channels
        self.image_size = image_size
        self.complex_data = complex_data

        layers = [initial_channels * 2**i for i in range(depth)]

        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        ReLU = partial(nn.LeakyReLU, negative_slope=0.2)
        Linear = nn.Linear

        self.inc = nn.Sequential(
            Conv2d(n_channels, layers[0], kernel_size=3, stride=1, padding=1),  # input is (3) x 96 x 96
            ReLU(inplace=True, n_channels=layers[0]) if complex_data else ReLU(inplace=True),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=2, padding=1, bias=False),  # state size. (64) x 48 x 48
            BatchNorm2d(layers[0]),
            ReLU(inplace=True, n_channels=layers[0]) if complex_data else ReLU(inplace=True),
        )

        features = []
        for i, layer in enumerate(layers[:-1]):
            features.extend(
                [
                    Conv2d(
                        layer,
                        layers[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),  # state size. (64) x 48 x 48
                    BatchNorm2d(layers[i + 1]),
                    ReLU(inplace=True, n_channels=layers[i + 1]) if complex_data else ReLU(inplace=True),
                    Conv2d(
                        layers[i + 1],
                        layers[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    BatchNorm2d(layers[i + 1]),
                    ReLU(inplace=True, n_channels=layers[i + 1]) if complex_data else ReLU(inplace=True),
                ]
            )

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            Linear(self._get_adv_shape(), 1024),
            ReLU(inplace=True, n_channels=1024) if complex_data else ReLU(inplace=True),
            Linear(1024, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["data"]

        out = self.inc(x)
        out = self.features(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.sigmoid(out.abs())

        return out

    def _get_adv_shape(self):
        with torch.no_grad():
            test_input = torch.randn((1, self.n_channels, self.image_size, self.image_size))

            if self.complex_data:
                test_input = torch.complex(test_input, test_input)
            output = self.features(self.inc(test_input))
        return prod(output.shape)
