"""
    Train a WGAN-GP on MNIST.
"""

import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.utils as vutils
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16

#According to paper
CRITIC_ITERS = 5
LAMBDA = 10


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
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

img_list = []

gen.train()
critic.train()
iters = 0

for epoch in range(NUM_EPOCHS):
    for i, (imgs, _) in enumerate(loader):
        iters += 1
        imgs = imgs.to(device) #real
        curr_batch_size = imgs.shape[0]

        #From paper train critic 5 iters for 1 iter of generator
        #Train critic: max E[critic(real)] - E[critic(fake)] i.e min -(E[critic(real)] - E[critic(fake)])


        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        # Train critic
        fake = gen(noise)

        real_outputs = critic(imgs).reshape(-1)
        fake_outputs = critic(fake.detach()).reshape(-1) # detach for reuse in generator
        gp = gradient_penalty(critic, imgs, fake, device= device)
        critic_loss = ( 
            -(torch.mean(real_outputs) - torch.mean(fake_outputs)) + LAMBDA * gp
            )
        critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        # Train Generator
        
        
        # Train Generator: max E[critic(gen_fake)] i.e min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        gen_loss = -torch.mean(gen_fake)
        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()


        if (iters % 200 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(loader)-1)):
            with torch.no_grad():
                faker = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(faker, padding=2, normalize=True))



        if i == 0:
            print("Epoch: {}/{}".format(epoch, NUM_EPOCHS))
            print("Critic loss: {}".format(d_loss))
            print("Generator loss: {}".format(g_loss))
            # print("Real outputs: {}".format(real_outputs))
            # print("Fake outputs: {}".format(fake_outputs))
            # print("Real labels: {}".format(real_labels))
            # print("Fake labels: {}".format(fake_labels))
            # print("Generator output: {}".format(gen(noise)))
            # print("Critic output: {}".format(critic(imgs)))
            print("\n")


fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

