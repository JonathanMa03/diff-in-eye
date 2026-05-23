import numpy as np
import pandas as pd
import torch

from .baselines import (
    zero_fill_reconstruction,
    gaussian_reconstruction,
    pde_diffusion_reconstruction,
    telea_reconstruction,
)
from .masks import horizontal_stripe_mask
from .metrics import masked_mse, masked_mae, psnr
from .preprocess import preprocess_image
from .diff_sampling import sample_conditional_inpaint


def ddpm_reconstruction(
    original,
    mask,
    model,
    alpha_bars,
    device,
    sample_steps=100,
):
    masked = original * mask

    masked_tensor = (
        torch.tensor(masked)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    mask_tensor = (
        torch.tensor(mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    recon = sample_conditional_inpaint(
        model=model,
        masked_image=masked_tensor,
        mask=mask_tensor,
        alpha_bars=alpha_bars,
        device=device,
        sample_steps=sample_steps,
        show_progress=False,
    )

    return recon[0, 0].detach().cpu().numpy()


def evaluate_reconstruction(
    original,
    reconstruction,
    mask,
    method_name,
):
    return {
        f"{method_name}_mse": masked_mse(original, reconstruction, mask),
        f"{method_name}_mae": masked_mae(original, reconstruction, mask),
        f"{method_name}_psnr": psnr(original, reconstruction),
    }


def evaluate_classical_methods(
    image_paths,
    n_samples=100,
    mask_fn=horizontal_stripe_mask,
    preprocess_fn=preprocess_image,
):
    rows = []

    for i, path in enumerate(image_paths[:n_samples]):
        original = preprocess_fn(path)
        mask = mask_fn(original.shape)
        masked = original * mask

        zero = zero_fill_reconstruction(masked, mask)
        gaussian = gaussian_reconstruction(masked, mask)
        pde = pde_diffusion_reconstruction(masked, mask)
        telea = telea_reconstruction(masked, mask)

        row = {
            "idx": i,
            "path": str(path),
        }

        row.update(evaluate_reconstruction(original, zero, mask, "zero"))
        row.update(evaluate_reconstruction(original, gaussian, mask, "gaussian"))
        row.update(evaluate_reconstruction(original, pde, mask, "pde"))
        row.update(evaluate_reconstruction(original, telea, mask, "telea"))

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_all_methods(
    image_paths,
    model,
    alpha_bars,
    device,
    n_samples=25,
    sample_steps=100,
    mask_fn=horizontal_stripe_mask,
    preprocess_fn=preprocess_image,
    verbose=True,
):
    rows = []

    model.eval()

    for i, path in enumerate(image_paths[:n_samples]):
        if verbose:
            print(f"Processing {i + 1}/{n_samples}")

        original = preprocess_fn(path)
        mask = mask_fn(original.shape)
        masked = original * mask

        zero = zero_fill_reconstruction(masked, mask)
        gaussian = gaussian_reconstruction(masked, mask)
        pde = pde_diffusion_reconstruction(masked, mask)
        telea = telea_reconstruction(masked, mask)
        ddpm = ddpm_reconstruction(
            original=original,
            mask=mask,
            model=model,
            alpha_bars=alpha_bars,
            device=device,
            sample_steps=sample_steps,
        )

        row = {
            "idx": i,
            "path": str(path),
        }

        row.update(evaluate_reconstruction(original, zero, mask, "zero"))
        row.update(evaluate_reconstruction(original, gaussian, mask, "gaussian"))
        row.update(evaluate_reconstruction(original, pde, mask, "pde"))
        row.update(evaluate_reconstruction(original, telea, mask, "telea"))
        row.update(evaluate_reconstruction(original, ddpm, mask, "ddpm"))

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(results_df):
    metric_cols = [
        col for col in results_df.columns
        if col.endswith("_mse") or col.endswith("_mae") or col.endswith("_psnr")
    ]

    return results_df[metric_cols].describe()


def method_summary(results_df):
    methods = sorted(
        set(col.rsplit("_", 1)[0] for col in results_df.columns if col.endswith("_mse"))
    )

    rows = []

    for method in methods:
        rows.append({
            "method": method,
            "mean_mse": results_df[f"{method}_mse"].mean(),
            "mean_mae": results_df[f"{method}_mae"].mean(),
            "mean_psnr": results_df[f"{method}_psnr"].mean(),
        })

    return pd.DataFrame(rows).sort_values("mean_mse")