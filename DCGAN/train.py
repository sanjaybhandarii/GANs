"""
    Train a DCGAN on MNIST.
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Generator, Discriminator, initialize_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64


transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],[ 0.5 for _ in range(CHANNELS_IMG)]),

]
)

dataset = datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1)

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        # Train Discriminator
        fake = gen(noise)

        real_labels = torch.ones(BATCH_SIZE, 1, 1, 1).to(device)
        fake_labels = torch.zeros(BATCH_SIZE, 1, 1, 1).to(device)
        real_outputs = disc(imgs)
        fake_outputs = disc(fake.detach()) # detach to avoid backprop through generator
        real_loss = criterion(real_outputs, torch.ones_like(real_outputs))
        fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))
        d_loss = (real_loss + fake_loss) / 2
        disc.zero_grad()
        d_loss.backward()
        opt_disc.step()
        # Train Generator
        
        
        gen_outputs = disc(fake)
        g_loss = criterion(gen_outputs, torch.ones_like(gen_outputs))
        gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
        if i % 100 == 0:
            print("Epoch: {}/{}".format(epoch, NUM_EPOCHS))
            print("Discriminator loss: {}".format(d_loss))
            print("Generator loss: {}".format(g_loss))
            print("Real outputs: {}".format(real_outputs))
            print("Fake outputs: {}".format(fake_outputs))
            print("Real labels: {}".format(real_labels))
            print("Fake labels: {}".format(fake_labels))
            print("Generator output: {}".format(gen(noise)))
            print("Discriminator output: {}".format(disc(imgs)))
            print("\n")

