import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import T
from .diff_schedules import q_sample


def masked_noise_loss(pred_noise, true_noise, mask):
    missing_region = 1.0 - mask
    return (((pred_noise - true_noise) ** 2) * missing_region).sum() / missing_region.sum()


def save_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    losses,
    step,
    extra=None,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }

    if extra is not None:
        payload.update(extra)

    torch.save(payload, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    device="cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint.get("step", 0)
    losses = checkpoint.get("losses", [])

    return model, optimizer, start_step, losses, checkpoint


def train_conditional_ddpm(
    model,
    train_loader,
    optimizer,
    alpha_bars,
    device,
    n_steps=1000,
    use_masked_loss=True,
    log_every=100,
    checkpoint_every=None,
    checkpoint_dir=None,
    checkpoint_name="conditional_ddpm",
    start_step=0,
    existing_losses=None,
    extra_checkpoint_info=None,
):
    model.train()

    losses = [] if existing_losses is None else list(existing_losses)
    data_iter = iter(train_loader)

    loss_fn = nn.MSELoss()

    pbar = tqdm(range(n_steps))

    for local_step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x0 = batch["image"].to(device)
        masked_image = batch["masked_image"].to(device)
        mask = batch["mask"].to(device)

        batch_size = x0.shape[0]

        t = torch.randint(
            0,
            T,
            (batch_size,),
            device=device,
        ).long()

        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, alpha_bars, noise)

        pred_noise = model(
            x_t,
            masked_image,
            mask,
            t,
        )

        if use_masked_loss:
            loss = masked_noise_loss(pred_noise, noise, mask)
        else:
            loss = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_step = start_step + local_step + 1
        losses.append(loss.item())

        if current_step % log_every == 0:
            pbar.set_description(
                f"Step {current_step} | Loss: {loss.item():.6f}"
            )

        if (
            checkpoint_every is not None
            and checkpoint_dir is not None
            and current_step % checkpoint_every == 0
        ):
            checkpoint_path = checkpoint_dir / f"{checkpoint_name}_step_{current_step}.pt"

            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                losses=losses,
                step=current_step,
                extra=extra_checkpoint_info,
            )

    return losses