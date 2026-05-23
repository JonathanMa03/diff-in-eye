from pathlib import Path
import argparse

import pandas as pd
import torch

from diffindeye.config import TEST_DIR, METRICS_DIR
from diffindeye.data import get_image_paths
from diffindeye.diff_models import ConditionalTimeDDPM
from diffindeye.diff_schedules import make_linear_schedule
from diffindeye.evaluation import (
    evaluate_all_methods,
    evaluate_classical_methods,
    ddpm_reconstruction,
    evaluate_reconstruction,
)
from diffindeye.masks import horizontal_stripe_mask
from diffindeye.preprocess import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline-vs-diffusion", "diffusion-vs-diffusion"],
        required=True,
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more DDPM checkpoint paths.",
    )

    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = ConditionalTimeDDPM().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def evaluate_diffusion_checkpoints(
    image_paths,
    checkpoints,
    alpha_bars,
    device,
    n_samples=25,
    sample_steps=100,
):
    rows = []

    for i, path in enumerate(image_paths[:n_samples]):
        print(f"Processing image {i + 1}/{n_samples}")

        original = preprocess_image(path)
        mask = horizontal_stripe_mask(original.shape)

        row = {
            "idx": i,
            "path": str(path),
        }

        for checkpoint_path in checkpoints:
            checkpoint_path = Path(checkpoint_path)
            run_label = checkpoint_path.stem

            print(f"  Evaluating {run_label}")

            model = load_model(checkpoint_path, device)

            ddpm = ddpm_reconstruction(
                original=original,
                mask=mask,
                model=model,
                alpha_bars=alpha_bars,
                device=device,
                sample_steps=sample_steps,
            )

            metrics = evaluate_reconstruction(
                original=original,
                reconstruction=ddpm,
                mask=mask,
                method_name=run_label,
            )

            row.update(metrics)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_method(results_df):
    methods = sorted(
        set(
            col.rsplit("_", 1)[0]
            for col in results_df.columns
            if col.endswith("_mse")
        )
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


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        run_name = args.mode
    else:
        run_name = args.run_name

    test_paths = get_image_paths(TEST_DIR)

    schedule = make_linear_schedule(device=device)
    alpha_bars = schedule["alpha_bars"]

    if args.mode == "baseline-vs-diffusion":
        if len(args.checkpoints) != 1:
            raise ValueError(
                "baseline-vs-diffusion mode expects exactly one checkpoint."
            )

        checkpoint_path = Path(args.checkpoints[0])
        model = load_model(checkpoint_path, device)

        results = evaluate_all_methods(
            image_paths=test_paths,
            model=model,
            alpha_bars=alpha_bars,
            device=device,
            n_samples=args.n_samples,
            sample_steps=args.sample_steps,
            verbose=True,
        )

    elif args.mode == "diffusion-vs-diffusion":
        results = evaluate_diffusion_checkpoints(
            image_paths=test_paths,
            checkpoints=args.checkpoints,
            alpha_bars=alpha_bars,
            device=device,
            n_samples=args.n_samples,
            sample_steps=args.sample_steps,
        )

    summary = summarize_by_method(results)

    output_path = METRICS_DIR / f"{run_name}_results.csv"
    summary_path = METRICS_DIR / f"{run_name}_summary.csv"

    results.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("Saved results to:", output_path)
    print("Saved summary to:", summary_path)
    print()
    print(summary)


if __name__ == "__main__":
    main()