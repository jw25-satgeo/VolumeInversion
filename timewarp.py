import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# Continuous Diffusion for Mixed-Type Tabular Data @ https://arxiv.org/html/2312.10431v1
#  src @  https://github.com/muellermarkus/cdtd/blob/main/layers/timewarping.py
class Timewarp_Logistic(nn.Module):
    """
    A version of timewarping with exact cdfs instead of p.w.l. functions.
    It uses a domain-adapted cdf of the logistic distribution.
    Adapted from : Continuous Diffusion for Mixed-Type Tabular Data (Mueller et al., 2023)
    """

    def __init__(self, num_features, sigma_min, sigma_max, weight_low_noise=1.0):
        super().__init__()

        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)
        self.num_funcs = num_features

        # init parameters
        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v - 1) - 1)
        self.logits_v = nn.Parameter(  torch.full((self.num_funcs,), fill_value=logit_v)  )
        self.register_buffer("init_v", self.logits_v.clone())

        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        self.logits_mu = nn.Parameter(  torch.full((self.num_funcs,), fill_value=logit_mu)  )
        self.register_buffer("init_mu", self.logits_mu.clone())

        # init gamma, scaling parameter to 1
        self.logits_gamma = nn.Parameter(  (torch.ones((self.num_funcs, 1)).exp() - 1).log()  )

    def get_params(self):
        logit_mu = self.logits_mu  # let underlying parameter be ln(mu / (1-mu))
        v = 1 + F.softplus(self.logits_v)  # v > 1
        scale = F.softplus(self.logits_gamma)
        return logit_mu, v, scale

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return 1 / (1 + Z)

    def pdf_fn(self, x, logit_mu, v):
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (Z / ((1 + Z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        c = logit_mu + 1 / v * torch.special.logit(u, eps=1e-7)
        return F.sigmoid(c)

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        # sigmas -> x : (n_sigmas, features)
        logit_mu, v, scale = self.get_params()
        if not invert:
            if normalize: scale = 1.0
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min) # can have more sigmas than cdfs
            x = torch.clamp(x, 1e-7, 1 - 1e-7) # ensure x is never 0 or 1 to ensure robustness
            if return_pdf: output = (  torch.vmap(self.pdf_fn, in_dims=0)(x.T, logit_mu, v)  ).T
            else:          output = (  torch.vmap(self.cdf_fn, in_dims=0)(x.T, logit_mu, v) * scale  ).T
        else:
            # have single u, need to repeat u
            inp = repeat(x, "b -> f b", f=self.num_funcs)
            output = (torch.vmap(self.quantile_fn, in_dims=0)(inp, logit_mu, v)).T
            output = output.masked_fill((x == 0.0).unsqueeze(-1), 0.0)
            output = output.masked_fill((x == 1.0).unsqueeze(-1), 1.0)
            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min

        return output

    def loss_fn(self, sigmas, losses):
        # losses and sigmas have shape (B, num_features)
        losses_estimated = self.forward(sigmas)
        with torch.no_grad(): pdf = self.forward(sigmas, return_pdf=True).detach()
        return ((losses_estimated - losses.detach()) ** 2) / (pdf + 1e-7)



        