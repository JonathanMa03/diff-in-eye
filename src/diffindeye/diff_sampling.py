import torch
from tqdm.auto import tqdm

from .config import T


@torch.no_grad()
def sample_conditional_inpaint(
    model,
    masked_image,
    mask,
    alpha_bars,
    device,
    sample_steps=100,
    show_progress=True,
    return_trajectory=False,
    save_every=None,
):
    model.eval()

    x = torch.randn_like(masked_image)
    x = mask * masked_image + (1.0 - mask) * x

    trajectory = []

    timesteps = torch.linspace(
        T - 1,
        0,
        sample_steps,
        device=device,
    ).long()

    iterator = range(len(timesteps) - 1)

    if show_progress:
        iterator = tqdm(iterator)

    for i in iterator:
        t = timesteps[i]
        t_prev = timesteps[i + 1]

        t_batch = torch.full(
            (x.shape[0],),
            t,
            device=device,
            dtype=torch.long,
        )

        pred_noise = model(
            x,
            masked_image,
            mask,
            t_batch,
        )

        alpha_bar_t = alpha_bars[t].view(1, 1, 1, 1)
        alpha_bar_prev = alpha_bars[t_prev].view(1, 1, 1, 1)

        x0_pred = (
            x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        ) / torch.sqrt(alpha_bar_t)

        x = (
            torch.sqrt(alpha_bar_prev) * x0_pred
            + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        )

        x = mask * masked_image + (1.0 - mask) * x
        x = x.clamp(0.0, 1.0)

        if return_trajectory:
            should_save = save_every is None or i % save_every == 0
            if should_save:
                trajectory.append(x.detach().cpu())

    if return_trajectory:
        return x, trajectory

    return x