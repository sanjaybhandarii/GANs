"""
    Discriminator and Generator based on DC-GAN paper
"""

from matplotlib import image
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d,num_classes,image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.disc = nn.Sequential(
            # input: N x img_channels x 64 x 64
            nn.Conv2d(
                img_channels+1, features_d, kernel_size=4, stride=2, padding=1
                #image_channels+1 for embedding layer of labels
            ), #32 x 32
            nn.LeakyReLU(0.2),

            self._block(features_d, features_d * 2, 4, 2, 1), #16 x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1), #8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1), #4 x 4
           
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

        self.embed = nn.Embedding(num_classes, image_size*image_size)

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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = (self.embed(labels)).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g, image_size, num_classes, embedding_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim+embedding_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x img_channels x 64 x 64
            nn.Tanh(),
        )

        self.embed = nn.Embedding(num_classes, embedding_size)

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

    def forward(self, x,labels):
        #latent vector z: N x z_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
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
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed")
