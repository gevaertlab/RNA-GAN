from math import ceil, log2

import torch.nn as nn
import torch.nn.functional as F

from torchgan.models import Discriminator, Generator

class DCGANUpGenerator(Generator):
    """Deep Convolutional Generator with resize convolution
    """
    def __init__(
        self,
        encoding_dims=100,
        out_size=32,
        out_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
    ):
        super(DCGANUpGenerator, self).__init__(encoding_dims, label_type)
        if out_size < 16 or ceil(log2(out_size)) != log2(out_size):
            raise Exception(
                "Target Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = out_size.bit_length() - 4
        self.ch = out_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        model = []
        d = int(self.n * (2 ** num_repeats))
        if batchnorm is True:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.encoding_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nn.BatchNorm2d(d),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor = 2, mode='bilinear'),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(d, d//2,
                                  kernel_size=3, stride=1, padding=0),
                        #nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm2d(d // 2),
                        nl,
                    )
                )
                d = d // 2
        else:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.encoding_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nl,
                    )
                )
                d = d // 2

        model.append(
            nn.Sequential(
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(d, self.ch,
                            kernel_size=3, stride=1, padding=0),
                #nn.ConvTranspose2d(d, self.ch, 4, 2, 1, bias=True), last_nl
            )
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()
    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the encoding ``x`` through the Generator.
        Args:
            x (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.
        Returns:
            A 4D torch.Tensor of the generated image.
        """
        x = x.view(-1, x.size(1), 1, 1)
        x = self.model(x)
        return x