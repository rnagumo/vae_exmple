
"""Principle Component Analysis"""


import torch
from torch.distributions import MultivariateNormal as MultivariateNormalTorch

import pixyz.distributions as pxd
from pixyz.distributions.distributions import DistributionBase


class MultivariateNormal(DistributionBase):

    @property
    def params_keys(self):
        return ["loc", "covariance_matrix"]

    @property
    def distribution_torch_class(self):
        return MultivariateNormalTorch

    @property
    def distribution_name(self):
        return "MultivariateNormal"


class Generator(pxd.Normal):

    def __init__(self, z_dim, x_dim):
        super().__init__(cond_var=["z"], var=["x"])

        self.x_dim = x_dim
        self.W = torch.randn(x_dim, z_dim)
        self.sigma = torch.ones(1)

    def forward(self, z):
        loc = z @ self.W.T
        scale = self.sigma
        return {"loc": loc, "scale": scale}

    def inference(self, x, q):
        """Input x size of (num_data, x_dim)

        z_expt: size of (num_data, z_dim)
        zz_expt: size of (z_dim, z_dim)

        Parameters
        ----------
        x : torch.tensor()
            size of (num_data, x_dim)
        q : VariationalPosterior
        """

        # Save original value
        W = self.W.clone()

        # Update W
        self.W = x.T @ q.z_expt @ q.zz_expt.inverse()

        # Update Sigma
        z_W_x = 0
        for n in range(q.z_expt.size(0)):
            z_W_x += q.z_expt[n].unsqueeze(0) @ W.T @ x[n].unsqueeze(1)
        z_W_x = z_W_x.squeeze()

        tr_zz_WW = torch.trace(q.zz_expt @ W.T @ W)

        self.sigma = ((x.size(0) * x.size(1)) ** -1
                      * (torch.trace(x @ x.T) - 2 * z_W_x + tr_zz_WW))


class Posterior(MultivariateNormal):

    def __init__(self, x_dim, generator):
        super().__init__(var=["x"])

        self.x_dim = x_dim
        self.generator = generator

    def forward(self):
        cov = (self.generator.sigma * torch.eye(self.x_dim)
               + self.generator.W @ self.generator.W.T)
        return {"loc": torch.zeros(self.x_dim), "covariance_matrix": cov}


class VarationalPosterior(MultivariateNormal):

    def __init__(self, z_dim, n_dim):
        super().__init__(var=["z"])

        # Dimension
        self.z_dim = z_dim
        self.n_dim = n_dim

        # Variational parameters
        self.mu = torch.zeros(n_dim, z_dim)
        self.Sigma = torch.eye(z_dim, z_dim)

        # Expectations
        self.z_expt = torch.randn(n_dim, z_dim)
        self.zz_expt = self.z_expt.T @ self.z_expt

    def forward(self):
        return {"loc": self.mu,
                "covariance_matrix": self.Sigma.expand(self.n_dim, -1, -1)}

    def inference(self, x, p):
        """Variational Inference

        Prior = N(z|0, I)
        """

        # Update variational parameters
        self.Sigma = (
            p.sigma ** -1 * p.W.T @ p.W + torch.eye(self.z_dim)).inverse()

        for n in range(self.n_dim):
            self.mu[n] = (p.sigma ** -1 * self.Sigma.inverse() @ p.W.T
                          @ x[n].unsqueeze(1)).T

        # Calculate expectations
        self.z_expt = self.mu.clone()
        self.zz_expt = self.Sigma + self.z_expt.T @ self.z_expt


class PCA:

    def __init__(self, x_dim, z_dim, n_dim):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_dim = n_dim

        # Generative model
        self.prior = pxd.Normal(var=["z"], features_shape=[z_dim],
                                loc=torch.tensor(0.), scale=torch.tensor(1.))
        self.generator = Generator(z_dim, x_dim)
        self.posterior = Posterior(x_dim, self.generator)

        # Variational model
        self.vposterior = VarationalPosterior(z_dim, n_dim)

    def sample(self, n_dim=None, variational=True):

        if n_dim is None:
            n_dim = self.n_dim

        if variational:
            # Sample from variatinal prior
            prior = self.vposterior
        else:
            # Standard Normal distribution
            prior = self.prior

        return (self.generator * prior).sample(batch_n=n_dim)

    def inference(self, x_dict, max_iter=50):

        evidence_list = []
        for _ in range(max_iter):
            # Maximize ELBO
            self.generator.inference(x_dict["x"], self.vposterior)

            # Minimize KL-divergence
            self.vposterior.inference(x_dict["x"], self.generator)

            evidence_list.append(self.evidence(x_dict))

        return evidence_list

    def evidence(self, x_dict):
        return self.posterior.get_log_prob(x_dict).mean()


if __name__ == "__main__":
    x_dim = 3
    z_dim = 2
    n_dim = 10
    pca = PCA(x_dim, z_dim, n_dim)
    x = torch.randn(n_dim, x_dim)

    print(pca.sample(variational=False))
    print(pca.sample())

    # Inference
    pca.inference({"x": x})
    print(pca.sample())

    # Model evidence
    print(pca.evidence({"x": x}))
