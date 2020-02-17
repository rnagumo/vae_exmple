
"""Variational Auto-Encoder"""


import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm


class Generator(pxd.Normal):

    def __init__(self, z_dim, h_dim, x_dim):
        super().__init__(cond_var=["z"], var=["x"])

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, x_dim)
        self.fc32 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        loc = self.fc31(h)
        scale = F.softplus(self.fc32(h))
        return {"loc": loc, "scale": scale}


class Inference(pxd.Normal):

    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(cond_var=["x"], var=["z"])

        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, z_dim)
        self.fc32 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        loc = self.fc31(h)
        scale = F.softplus(self.fc32(h))
        return {"loc": loc, "scale": scale}


def load_vae_model(x_dim, z_dim, h_dim):

    # Generative model
    prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim])
    decoder = Generator(z_dim, h_dim, x_dim)

    # Variational model
    encoder = Inference(x_dim, h_dim, z_dim)

    # Loss
    ce = pxl.CrossEntropy(encoder, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    loss = (ce + kl).mean()

    # Model
    vae = pxm.Model(loss, distributions=[encoder, decoder])

    return vae, (prior, decoder, encoder)


if __name__ == "__main__":
    x_dim = 3
    z_dim = 2
    h_dim = 5
    x = torch.randn(10, x_dim)

    vae, sampler = load_vae_model(x_dim, z_dim, h_dim)

    # Training
    for _ in range(3):
        vae.train({"x": x})

    # Reconstruction
    with torch.no_grad():
        prior, p, q = sampler
        z = q.sample({"x": x}, return_all=False)
        x_recon = p.sample_mean(z)
        print(x_recon.size(), x_recon[0])

    # Sample from latent
    with torch.no_grad():
        prior, p, q = sampler
        z_sample = prior.sample(batch_n=100)
        x_sample = p.sample_mean({"z": z_sample["z"]})
        print(x_sample.size(), x_sample[0])
