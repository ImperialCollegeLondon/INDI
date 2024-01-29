from torch import nn

from extensions.uformer_tensor_denoising.utils import instantiate


class GANUFormerModelWrapper(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = instantiate(generator)
        self.discriminator = instantiate(discriminator)

    def init_weights(self):
        ...

    def parameters(self, **kwargs):
        return {
            "generator": self.generator.parameters(),
            "discriminator": self.discriminator.parameters(),
        }
