#!/usr/bin/env python3
"""
Fine-tune a score-following model to predict music cursor positions from audio.

Two model backends are supported (set Config.model_type):
  "phi4"     — Phi-4-multimodal-instruct with speech-LoRA fine-tuning (~5.6 B params)
  "baseline" — Whisper-small + CLIP ViT-B/32 + fusion transformer (~250 M params)

The model takes:
  - Sheet music images (all pages)
  - A start position (x_ratio, y_ratio) on the image
  - An audio segment starting at the corresponding timestamp
And predicts three things at every output token:
  - pos_head:  patch within current page (grid_w × grid_h classes)
  - page_head: which page               (max_num_images classes)
  - bar_head:  rest-bar index           (max_bar classes, 0 = playing)
"""

import math
import os
import random
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ScoreFollowingDataset, PieceBatchSampler
from model import score_following_loss


def _trainable_state(model):
    """Return state_dict containing only parameters with requires_grad=True."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Model backend: "phi4" or "baseline"
    model_type = "phi4"
    model_name = "microsoft/Phi-4-multimodal-instruct"   # used by phi4 only

    # Audio
    audio_sample_rate = 16000       # Phi-4 and Whisper both expect 16 kHz

    # Prediction
    audio_length_sec = 20.0         # audio chunk length fed to the model
    sample_shift_sec = 5.0          # shift between consecutive training samples
    max_num_images   = 2            # max number of pages/images supported
    image_width      = 256          # images resized to this width before encoding
    pos_num_freqs    = 8            # Fourier frequency bands for (x, y) encoding
    grid_w           = 32           # patch grid width  per page
    grid_h           = 32           # patch grid height per page
    max_bar          = 64           # bar classes: 0=playing, 1..max_bar-1=rest bars

    # Training
    batch_size       = 32
    learning_rate    = 1e-4
    weight_decay     = 0.01
    num_epochs       = 50
    warmup_steps     = 15
    grad_accum_steps = 1
    max_grad_norm    = 1.0
    log_every_n_steps = -1

    # Data
    train_dirs = ["data/train/*"]
    dev_dirs   = ["data/dev/*"]     # if empty, 10% of train is held out

    # Output
    output_dir         = "checkpoints"
    save_every_n_epochs = -1


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
        n_val   = max(1, len(train_dataset) // 10)
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Dev samples:   {len(val_dataset)} (auto split from train)")

    # ── Create model ──────────────────────────────────────────────────────────
    if config.model_type == "baseline":
        from baseline_model import BaselineScoreFollowingModel
        model = BaselineScoreFollowingModel(config)
        model = model.to(device)
    else:
        from model import ScoreFollowingModel
        model = ScoreFollowingModel(config)   # handles device_map internally

    _collate = model.get_collate_fn(config.audio_sample_rate)

    train_sampler = PieceBatchSampler(
        train_dataset, config.batch_size, shuffle=True, drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(config.learning_rate)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in param_groups],
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    os.makedirs(config.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0

    def _ppl(ce):
        return math.exp(min(ce, 20))   # cap to avoid overflow on random init

    for epoch in range(config.num_epochs):
        model.train()
        # per-step accumulators
        e_pos_ce = e_page_ce = e_bar_ce = 0.0
        e_pos_acc = e_page_acc = e_bar_acc = 0.0
        e_steps = 0
        w_pos_ce = w_page_ce = w_bar_ce = 0.0
        w_pos_acc = w_page_acc = w_bar_acc = w_steps = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            start_pos        = batch.pop("start_pos").to(device)
            start_img        = batch.pop("start_img").to(device)
            start_bar        = batch.pop("start_bar").to(device)
            target_pos_patch = batch.pop("target_pos_patch").to(device)
            target_page      = batch.pop("target_page").to(device)
            target_bar       = batch.pop("target_bar").to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pos_logits, page_logits, bar_logits = model(inputs, start_pos, start_img, start_bar)
                loss, pos_loss, page_loss, bar_loss, pos_acc, page_acc, bar_acc = score_following_loss(
                    pos_logits, page_logits, bar_logits,
                    target_pos_patch, target_page, target_bar,
                )
                loss = loss / config.grad_accum_steps

            loss.backward()
            e_pos_ce   += pos_loss.item()
            e_page_ce  += page_loss.item()
            e_bar_ce   += bar_loss.item()
            e_pos_acc  += pos_acc.item()
            e_page_acc += page_acc.item()
            e_bar_acc  += bar_acc.item()
            e_steps    += 1

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                w_pos_ce   += pos_loss.item()
                w_page_ce  += page_loss.item()
                w_bar_ce   += bar_loss.item()
                w_pos_acc  += pos_acc.item()
                w_page_acc += page_acc.item()
                w_bar_acc  += bar_acc.item()
                w_steps    += 1

                if config.log_every_n_steps > 0 and global_step % config.log_every_n_steps == 0:
                    print(
                        f"  step {global_step}"
                        f"  pos_ppl={_ppl(w_pos_ce/w_steps):.1f}({w_pos_acc/w_steps:.3f})"
                        f"  page_ppl={_ppl(w_page_ce/w_steps):.2f}({w_page_acc/w_steps:.3f})"
                        f"  bar_ppl={_ppl(w_bar_ce/w_steps):.2f}({w_bar_acc/w_steps:.3f})"
                        f"  lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    w_pos_ce = w_page_ce = w_bar_ce = 0.0
                    w_pos_acc = w_page_acc = w_bar_acc = w_steps = 0.0

        t_pos_ce   = e_pos_ce   / e_steps
        t_page_ce  = e_page_ce  / e_steps
        t_bar_ce   = e_bar_ce   / e_steps
        t_pos_acc  = e_pos_acc  / e_steps
        t_page_acc = e_page_acc / e_steps
        t_bar_acc  = e_bar_acc  / e_steps

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        v_pos_ce = v_page_ce = v_bar_ce = 0.0
        v_pos_acc = v_page_acc = v_bar_acc = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                start_pos        = batch.pop("start_pos").to(device)
                start_img        = batch.pop("start_img").to(device)
                start_bar        = batch.pop("start_bar").to(device)
                target_pos_patch = batch.pop("target_pos_patch").to(device)
                target_page      = batch.pop("target_page").to(device)
                target_bar       = batch.pop("target_bar").to(device)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pos_logits, page_logits, bar_logits = model(inputs, start_pos, start_img, start_bar)
                    _, pos_loss, page_loss, bar_loss, pos_acc, page_acc, bar_acc = score_following_loss(
                        pos_logits, page_logits, bar_logits,
                        target_pos_patch, target_page, target_bar,
                    )
                v_pos_ce   += pos_loss.item()
                v_page_ce  += page_loss.item()
                v_bar_ce   += bar_loss.item()
                v_pos_acc  += pos_acc.item()
                v_page_acc += page_acc.item()
                v_bar_acc  += bar_acc.item()

        n_val = len(val_loader)
        vt_pos_ce   = v_pos_ce   / n_val
        vt_page_ce  = v_page_ce  / n_val
        vt_bar_ce   = v_bar_ce   / n_val
        vt_pos_acc  = v_pos_acc  / n_val
        vt_page_acc = v_page_acc / n_val
        vt_bar_acc  = v_bar_acc  / n_val
        avg_val_loss = vt_pos_ce + vt_page_ce + vt_bar_ce

        print(
            f"Epoch {epoch+1}/{config.num_epochs}\n"
            f"  train  pos={_ppl(t_pos_ce):7.1f}ppl/{t_pos_acc:.3f}acc"
            f"  page={_ppl(t_page_ce):6.2f}ppl/{t_page_acc:.3f}acc"
            f"  bar={_ppl(t_bar_ce):6.2f}ppl/{t_bar_acc:.3f}acc\n"
            f"  val    pos={_ppl(vt_pos_ce):7.1f}ppl/{vt_pos_acc:.3f}acc"
            f"  page={_ppl(vt_page_ce):6.2f}ppl/{vt_page_acc:.3f}acc"
            f"  bar={_ppl(vt_bar_ce):6.2f}ppl/{vt_bar_acc:.3f}acc"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                "epoch":                epoch,
                "trainable_state_dict": _trainable_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             best_val_loss,
                "config":               vars(config),
                "model_type":           config.model_type,
            }, save_path)
            print(f"  -> Saved best model (val_total_ce={best_val_loss:.4f})")

        if config.save_every_n_epochs > 0 and (epoch + 1) % config.save_every_n_epochs == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch":                epoch,
                "trainable_state_dict": _trainable_state(model),
                "val_loss":             avg_val_loss,
            }, save_path)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score following model training")
    parser.add_argument("--mode",       choices=["train", "export"], default="train")
    parser.add_argument("--model-type", choices=["phi4", "baseline"], default="baseline")
    parser.add_argument("--checkpoint", type=str,  help="Path to checkpoint for export")
    parser.add_argument("--output-dir", type=str,  default="onnx_export")
    parser.add_argument("--train-dirs", nargs="+",  default=["data/train/*"])
    parser.add_argument("--dev-dirs",   nargs="+",  default=["data/dev/*"])
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--lr",         type=float, default=1e-4)

    args = parser.parse_args()

    config = Config()
    config.model_type    = args.model_type
    config.train_dirs    = args.train_dirs
    config.dev_dirs      = args.dev_dirs
    config.num_epochs    = args.epochs
    config.learning_rate = args.lr

    if args.mode == "train":
        train(config)
    elif args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for export")
        from model import export_onnx
        export_onnx(config, args.checkpoint, args.output_dir)
