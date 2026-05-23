import torch

from .config import T, BETA_START, BETA_END


def make_linear_schedule(
    t_steps=T,
    beta_start=BETA_START,
    beta_end=BETA_END,
    device=None,
):
    betas = torch.linspace(beta_start, beta_end, t_steps)

    if device is not None:
        betas = betas.to(device)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


def q_sample(x0, t, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bars[t]).view(-1, 1, 1, 1)

    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise