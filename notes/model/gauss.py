
"""Gaussian Distribution"""


import torch

import pixyz.distributions as pxd


class Gaussian(pxd.Normal):

    def __init__(self, x_dim):
        super().__init__(var=["x"])

        self.mu = torch.zeros(x_dim)
        self.Sigma = torch.eye(x_dim)

    def forward(self):
        return {"loc": self.mu, "scale": self.Sigma}

    def inference(self, x):
        self.mu = x.mean(dim=0)
        self.Sigma = x.std(dim=0, unbiased=False)
