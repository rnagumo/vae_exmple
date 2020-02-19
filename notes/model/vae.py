
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


class VAE(pxm.Model):
    def __init__(self, x_dim, z_dim, h_dim):

        # Generative model
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim])
        self.decoder = Generator(z_dim, h_dim, x_dim)

        # Variational model
        self.encoder = Inference(x_dim, h_dim, z_dim)

        # Loss
        ce = pxl.CrossEntropy(self.encoder, self.decoder)
        kl = pxl.KullbackLeibler(self.encoder, self.prior)
        loss = (ce + kl).mean()

        # Init
        super().__init__(loss, distributions=[self.encoder, self.decoder])

    def reconstruction(self, x, return_all=False):
        with torch.no_grad():
            z = self.encoder.sample({"x": x}, return_all=False)
            x_recon = self.decoder.sample_mean(z)

        sample = {"x": x_recon}
        if return_all:
            sample.update(z)

        return sample

    def sample(self, sample_num, return_all=False):
        with torch.no_grad():
            z_sample_dict = self.prior.sample(batch_n=sample_num)
            x_sample = self.decoder.sample_mean(z_sample_dict)

        sample = {"x": x_sample}
        if return_all:
            sample.update(z_sample_dict)

        return sample


if __name__ == "__main__":
    x_dim = 3
    z_dim = 2
    h_dim = 5
    x = torch.randn(10, x_dim)

    vae = VAE(x_dim, z_dim, h_dim)

    # Training
    for _ in range(3):
        vae.train({"x": x})

    # Reconstruction
    recon = vae.reconstruction(x, return_all=True)
    print(recon["z"].size(), recon["z"][0])
    print(recon["x"].size(), recon["x"][0])

    # Sample from latent
    sample = vae.sample(1, return_all=True)
    print(sample["x"].size(), sample["x"][0])
    print(sample["z"].size(), sample["z"][0])
