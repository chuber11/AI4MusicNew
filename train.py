#!/usr/bin/env python3
"""
Fine-tune Phi-4-multimodal-instruct to predict music score positions from audio.

The model takes:
  - A sheet music image (all pages)
  - A start position (x_ratio, y_ratio) on the image
  - An audio segment starting at the corresponding timestamp
And predicts the cursor position at every LLM token output position.

Architecture: Phi-4-multimodal-instruct backbone with speech-LoRA fine-tuned,
plus a lightweight regression head for coordinate + page prediction.
"""

import math
import os
import random
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ScoreFollowingDataset, collate_fn
from model import ScoreFollowingModel, export_onnx, score_following_loss


def _trainable_state(model):
    """Return state_dict containing only parameters with requires_grad=True."""
    return {k: v for k, v in model.state_dict().items()
            if model.get_parameter(k).requires_grad}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Model
    model_name = "microsoft/Phi-4-multimodal-instruct"

    # Audio
    audio_sample_rate = 16000       # Phi-4 audio processor expects 16kHz

    # Prediction
    audio_length_sec = 10.0         # audio chunk length fed to the model (seconds)
    sample_shift_sec = 1.0          # shift between consecutive training samples (seconds)
    max_num_images = 10             # max number of pages/images supported
    pos_num_freqs = 8               # Fourier frequency bands for (x, y) encoding

    # Training
    batch_size = 2
    learning_rate = 1e-4
    weight_decay = 0.01
    num_epochs = 50
    warmup_ratio = 0.1
    grad_accum_steps = 4
    max_grad_norm = 1.0

    # Data
    train_dirs = ["data/train/*"]
    dev_dirs   = ["data/dev/*"]     # if empty, 10% of train data is held out automatically

    # Output
    output_dir = "checkpoints"
    save_every_n_epochs = 5


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(config=None):
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ScoreFollowingDataset(config.train_dirs, config)
    print(f"Train samples: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found. Check --train-dirs.")

    if config.dev_dirs:
        val_dataset = ScoreFollowingDataset(config.dev_dirs, config)
        print(f"Dev samples:   {len(val_dataset)}")
        if len(val_dataset) == 0:
            raise RuntimeError("No dev samples found. Check --dev-dirs.")
    else:
        n_val = max(1, len(train_dataset) // 10)
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Dev samples:   {len(val_dataset)} (auto split from train)")

    # Model must be created before DataLoaders so we can pass its processor
    model = ScoreFollowingModel(config)

    _collate = partial(collate_fn,
                       processor=model.processor,
                       audio_sample_rate=config.audio_sample_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer: speech-LoRA at lower LR; projection + head at full LR
    speech_lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    new_params = (
        list(model.pos_proj.parameters())
        + list(model.page_proj.parameters())
        + list(model.head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": speech_lora_params, "lr": config.learning_rate * 0.1},
        {"params": new_params,         "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.learning_rate * 0.1, config.learning_rate],
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="cos",
    )

    os.makedirs(config.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    print("Training starts now.")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        window_loss  = 0.0
        window_mse   = 0.0
        window_ce    = 0.0
        window_steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            start_pos  = batch.pop("start_pos").to(device)
            start_img  = batch.pop("start_img").to(device)
            target_xy  = batch.pop("target_xy").to(device)
            target_img = batch.pop("target_img").to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_xy, img_logits = model(inputs, start_pos, start_img)
                loss, coord_loss, img_loss = score_following_loss(
                    pred_xy, img_logits, target_xy, target_img,
                )
                loss = loss / config.grad_accum_steps

            loss.backward()
            step_loss = loss.item() * config.grad_accum_steps
            epoch_loss += step_loss

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                window_loss  += step_loss
                window_mse   += coord_loss.item()
                window_ce    += img_loss.item()
                window_steps += 1

                if global_step % 10 == 0:
                    avg_mse = window_mse / window_steps
                    avg_ce  = window_ce  / window_steps
                    print(f"  step {global_step}"
                          f"  loss={window_loss / window_steps:.6f}"
                          f"  mse={avg_mse:.6f}"
                          f"  ce={avg_ce:.4f}  ppl={math.exp(avg_ce):.2f}"
                          f"  lr={scheduler.get_last_lr()[0]:.2e}")
                    window_loss  = 0.0
                    window_mse   = 0.0
                    window_ce    = 0.0
                    window_steps = 0

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                start_pos  = batch.pop("start_pos").to(device)
                start_img  = batch.pop("start_img").to(device)
                target_xy  = batch.pop("target_xy").to(device)
                target_img = batch.pop("target_img").to(device)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_xy, img_logits = model(inputs, start_pos, start_img)
                    loss, _, _ = score_following_loss(
                        pred_xy, img_logits, target_xy, target_img,
                    )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config.num_epochs}  "
              f"train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "trainable_state_dict": _trainable_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config),
            }, save_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "trainable_state_dict": _trainable_state(model),
                "val_loss": avg_val_loss,
            }, save_path)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score following model training")
    parser.add_argument("--mode", choices=["train", "export"], default="train")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for export")
    parser.add_argument("--output-dir", type=str, default="onnx_export")
    parser.add_argument("--train-dirs", nargs="+", default=["data/train/*"])
    parser.add_argument("--dev-dirs",   nargs="+", default=["data/dev/*"])
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--audio-length", type=float, default=10.0,
                        help="Audio chunk length in seconds")
    parser.add_argument("--sample-shift", type=float, default=1.0,
                        help="Shift between consecutive training samples (seconds)")

    args = parser.parse_args()

    config = Config()
    config.train_dirs       = args.train_dirs
    config.dev_dirs         = args.dev_dirs
    config.num_epochs       = args.epochs
    config.batch_size       = args.batch_size
    config.learning_rate    = args.lr
    config.audio_length_sec = args.audio_length
    config.sample_shift_sec = args.sample_shift

    if args.mode == "train":
        train(config)
    elif args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for export")
        export_onnx(config, args.checkpoint, args.output_dir)
