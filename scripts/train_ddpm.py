from pathlib import Path
from datetime import datetime
import argparse
import json
import random

import numpy as np
import torch

from diffindeye.config import (
    TRAIN_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    SEED,
    T,
    BETA_START,
    BETA_END,
    TARGET_SIZE,
    STRIPE_HEIGHT,
)
from diffindeye.data import get_image_paths, make_dataloader
from diffindeye.diff_models import ConditionalTimeDDPM, ConditionalUNetDDPM
from diffindeye.diff_schedules import make_linear_schedule
from diffindeye.diff_train import (
    train_conditional_ddpm,
    save_training_checkpoint,
    load_training_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="conditional_ddpm")
    parser.add_argument("--model", type=str, choices=["simple", "unet"], default="simple")

    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time-emb-dim", type=int, default=32)

    parser.add_argument("--use-masked-loss", action="store_true")
    parser.add_argument("--no-masked-loss", dest="use_masked_loss", action="store_false")
    parser.set_defaults(use_masked_loss=True)

    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name, time_emb_dim, device):
    if model_name == "simple":
        return ConditionalTimeDDPM(
            time_emb_dim=time_emb_dim
        ).to(device)

    if model_name == "unet":
        return ConditionalUNetDDPM(
            time_emb_dim=time_emb_dim
        ).to(device)

    raise ValueError(f"Unknown model name: {model_name}")


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{args.model}_{timestamp}"

    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DDPM Training Run")
    print("=" * 70)
    print(f"Run ID:        {run_id}")
    print(f"Device:        {device}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Run directory: {run_dir}")
    print()

    print("Configuration")
    print("-" * 70)
    print(f"model:            {args.model}")
    print(f"n_steps:          {args.n_steps}")
    print(f"batch_size:       {args.batch_size}")
    print(f"learning_rate:    {args.lr}")
    print(f"time_emb_dim:     {args.time_emb_dim}")
    print(f"use_masked_loss:  {args.use_masked_loss}")
    print(f"checkpoint_every: {args.checkpoint_every}")
    print(f"log_every:        {args.log_every}")
    print(f"seed:             {args.seed}")
    print(f"target_size:      {TARGET_SIZE}")
    print(f"stripe_height:    {STRIPE_HEIGHT}")
    print(f"T:                {T}")
    print(f"beta_start:       {BETA_START}")
    print(f"beta_end:         {BETA_END}")
    print()

    train_paths = get_image_paths(TRAIN_DIR)

    print("Data")
    print("-" * 70)
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Train images:    {len(train_paths)}")
    print()

    train_loader = make_dataloader(
        image_paths=train_paths,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    batch = next(iter(train_loader))

    print("Batch sanity check")
    print("-" * 70)
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: {value.shape}")
    print()

    schedule = make_linear_schedule(
        t_steps=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    alpha_bars = schedule["alpha_bars"]

    print("Schedule sanity check")
    print("-" * 70)
    print(f"alpha_bars shape: {alpha_bars.shape}")
    print(f"alpha_bar[0]:     {alpha_bars[0].item():.6f}")
    print(f"alpha_bar[-1]:    {alpha_bars[-1].item():.6f}")
    print()

    model = build_model(
        model_name=args.model,
        time_emb_dim=args.time_emb_dim,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    start_step = 0
    existing_losses = []

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)

        print("Resuming from checkpoint")
        print("-" * 70)
        print(f"Checkpoint: {resume_path}")

        model, optimizer, start_step, existing_losses, checkpoint = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        print(f"Resumed from step: {start_step}")
        print(f"Previous losses:   {len(existing_losses)}")
        print()

    n_params = sum(p.numel() for p in model.parameters())

    print("Model")
    print("-" * 70)
    print(model)
    print()
    print(f"Parameters: {n_params:,}")
    print()

    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model": args.model,
        "device": str(device),
        "torch_version": torch.__version__,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "time_emb_dim": args.time_emb_dim,
        "use_masked_loss": args.use_masked_loss,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "seed": args.seed,
        "target_size": TARGET_SIZE,
        "stripe_height": STRIPE_HEIGHT,
        "T": T,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "resume_from": args.resume_from,
        "start_step": start_step,
        "n_train_images": len(train_paths),
        "n_params": n_params,
    }

    config_path = run_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("Saved config to:", config_path)
    print()

    print("Starting training")
    print("=" * 70)

    losses = train_conditional_ddpm(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        alpha_bars=alpha_bars,
        device=device,
        n_steps=args.n_steps,
        use_masked_loss=args.use_masked_loss,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=run_dir,
        checkpoint_name=args.run_name,
        start_step=start_step,
        existing_losses=existing_losses,
        extra_checkpoint_info=config,
    )

    final_step = start_step + args.n_steps
    final_checkpoint_path = run_dir / f"{args.run_name}_{args.model}_final_step_{final_step}.pt"

    save_training_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        losses=losses,
        step=final_step,
        extra=config,
    )

    print()
    print("=" * 70)
    print("Training complete")
    print("=" * 70)
    print(f"Final step:        {final_step}")
    print(f"Final checkpoint:  {final_checkpoint_path}")
    print(f"Config:            {config_path}")

    if len(losses) > 0:
        print(f"Final loss:        {losses[-1]:.6f}")
        print(f"Min loss:          {min(losses):.6f}")
        print(f"Mean last 50 loss: {np.mean(losses[-50:]):.6f}")

    print()
    print("Next evaluation command:")
    print(
        "PYTHONPATH=src python scripts/evaluate_methods.py "
        "--mode baseline-vs-diffusion "
        f"--checkpoints \"{final_checkpoint_path}\" "
        "--n-samples 25 "
        "--sample-steps 100 "
        f"--run-name {run_id}"
    )


if __name__ == "__main__":
    main()