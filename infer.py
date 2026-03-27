#!/usr/bin/env python3
"""
Run the score-following model on a data directory and produce an annotation JSON
compatible with create_verification_video.py.

The full audio (from --start-sec onwards) is passed to the model in one shot.
The starting cursor position is interpolated from the ground-truth annotation.

Usage:
    python infer.py --data-dir data/MyPiece --checkpoint checkpoints/best_model.pt
    python infer.py --data-dir data/MyPiece --checkpoint checkpoints/best_model.pt \\
                    --start-sec 30 --output predictions.json --interval 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import detect_lines, build_interpolation, get_position_at_time
from model import ScoreFollowingModel
from train import Config


def infer(config, checkpoint_path, data_dir, output_path,
          start_sec=0.0, annotation_interval_ms=100):
    data_dir = Path(data_dir)

    # ── Load annotation file ──────────────────────────────────────────────────
    ann_files = list(data_dir.glob("annotations_*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotations_*.json found in {data_dir}")
    with open(ann_files[0]) as f:
        existing = json.load(f)

    audio_filename  = existing["audio_filename"]
    image_filenames = existing["image_filenames"]
    audio_path      = data_dir / audio_filename

    # ── Interpolate start position from ground-truth annotation ───────────────
    lines     = detect_lines(existing)
    line_info = build_interpolation(lines)
    start_ms  = start_sec * 1000.0
    img_idx_start, x_start, y_start = get_position_at_time(start_ms, line_info)
    print(f"Start position at {start_sec:.1f}s: "
          f"page={img_idx_start}  x={x_start:.3f}  y={y_start:.3f}")

    # ── Load audio ────────────────────────────────────────────────────────────
    try:
        import soundfile as sf
        audio_data, sr = sf.read(str(audio_path))
    except Exception:
        import librosa
        audio_data, sr = librosa.load(str(audio_path), sr=config.audio_sample_rate, mono=True)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != config.audio_sample_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=config.audio_sample_rate)

    audio_duration_ms = len(audio_data) / config.audio_sample_rate * 1000

    # Trim to [start_sec, end]
    start_sample = int(start_sec * config.audio_sample_rate)
    audio_data   = audio_data[start_sample:].astype(np.float32)
    infer_duration_ms = len(audio_data) / config.audio_sample_rate * 1000

    # ── Load and resize images ────────────────────────────────────────────────
    all_images = []
    for fname in image_filenames:
        img = Image.open(str(data_dir / fname)).convert("RGB")
        w, h = img.size
        img = img.resize((config.image_width, int(h * config.image_width / w)), Image.LANCZOS)
        all_images.append(img)

    num_pages = len(all_images)
    prompt = "".join(f"<|image_{j+1}|>" for j in range(num_pages)) + "<|pos|><|page|><|audio_1|>"

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = ScoreFollowingModel(config)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key   = "trainable_state_dict" if "trainable_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key], strict=False)
    model.eval()

    device = next(model.backbone.parameters()).device

    start_pos = torch.tensor([[x_start, y_start]], dtype=torch.float32, device=device)
    start_img = torch.tensor([img_idx_start],      dtype=torch.long,    device=device)

    # ── Single forward pass on the full audio ─────────────────────────────────
    print(f"Running inference on {infer_duration_ms/1000:.1f}s of audio...")
    inputs = model.processor(
        text=[prompt],
        images=all_images,
        audios=[(audio_data, config.audio_sample_rate)],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_xy, img_logits = model(inputs, start_pos, start_img)

    pred_xy   = pred_xy[0].float().cpu()                # (seq_len, 2)
    pred_page = img_logits[0].float().cpu().argmax(-1)  # (seq_len,)
    seq_len   = pred_xy.shape[0]
    print(f"Model output: {seq_len} tokens")

    # ── Map token predictions to timestamps ───────────────────────────────────
    # Tokens are treated as uniformly covering [start_ms, start_ms + infer_duration_ms]
    annotations = []
    t_ms = start_ms
    while t_ms <= start_ms + infer_duration_ms:
        frac    = (t_ms - start_ms) / infer_duration_ms
        tok_idx = min(int(frac * seq_len), seq_len - 1)

        annotations.append({
            "timestamp_ms": round(t_ms),
            "x_ratio":      float(pred_xy[tok_idx, 0]),
            "y_ratio":      float(pred_xy[tok_idx, 1]),
            "image_index":  int(pred_page[tok_idx]),
        })
        t_ms += annotation_interval_ms

    # ── Write output ──────────────────────────────────────────────────────────
    result = {
        "audio_filename":    audio_filename,
        "image_filenames":   image_filenames,
        "audio_duration_ms": audio_duration_ms,
        "annotations":       annotations,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(annotations)} annotations → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True,
                        help="Directory with audio, images, and annotations_*.json")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output",     default=None,
                        help="Output JSON path (default: <data-dir>/predictions.json)")
    parser.add_argument("--start-sec",  type=float, default=0.0,
                        help="Start time in seconds; cursor position is taken from "
                             "the ground-truth annotation at that time (default: 0)")
    parser.add_argument("--interval",   type=float, default=100,
                        help="Annotation interval in ms (default: 100)")
    args = parser.parse_args()

    output = args.output or str(Path(args.data_dir) / "predictions.json")

    config = Config()
    infer(config, args.checkpoint, args.data_dir, output,
          start_sec=args.start_sec, annotation_interval_ms=args.interval)
