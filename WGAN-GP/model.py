"""
Critic and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            # input: N x img_channels x 64 x 64
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ), #32 x 32
            nn.LeakyReLU(0.2),

            self._block(features_d, features_d * 2, 4, 2, 1), #16 x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1), #8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1), #4 x 4
           
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # 1 x 1
            
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x img_channels x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02) # mean 0 std 0.02

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    critic = Critic(in_channels, 8)
    assert critic(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed")

