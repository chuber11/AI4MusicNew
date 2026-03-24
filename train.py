#!/usr/bin/env python3
"""
Fine-tune Phi-4-multimodal-instruct to predict music score positions from audio.

The model takes:
  - A sheet music image
  - A start position (x_ratio, y_ratio) on the image
  - An audio segment starting at the corresponding timestamp
And predicts the next N positions as the music progresses.

Architecture: Phi-4-multimodal-instruct is used as the backbone. We add a lightweight
regression head that replaces text generation with coordinate prediction.
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Model
    model_name = "microsoft/Phi-4-multimodal-instruct"

    # Audio
    audio_sample_rate = 16000       # Phi-4 audio processor expects 16kHz

    # Prediction
    pred_steps = 10                 # number of future positions to predict
    pred_interval_ms = 500          # interval between predicted positions (ms)
    sample_shift_ms = 250           # shift between training samples (ms)
    max_num_images = 10             # max number of pages/images supported

    @property
    def audio_segment_sec(self):
        """Inferred from pred_steps * pred_interval_ms."""
        return self.pred_steps * self.pred_interval_ms / 1000.0

    # Training
    batch_size = 2
    learning_rate = 1e-4
    weight_decay = 0.01
    num_epochs = 50
    warmup_ratio = 0.1
    grad_accum_steps = 4
    max_grad_norm = 1.0

    # LoRA (for efficient fine-tuning)
    use_lora = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # Data
    data_dirs = ["data/Hands_Across_the_Sea"]

    # Output
    output_dir = "checkpoints"
    save_every_n_epochs = 5


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def detect_lines(annotations):
    """Detect music lines by finding where x_ratio drops or y_ratio jumps."""
    anns = annotations["annotations"]
    lines = []
    current_line = [anns[0]]
    for i in range(1, len(anns)):
        a = anns[i]
        prev = anns[i - 1]
        x_drops = a["x_ratio"] < prev["x_ratio"] - 0.2
        y_jumps = abs(a["y_ratio"] - prev["y_ratio"]) > 0.05
        img_changes = a["image_index"] != prev["image_index"]
        if x_drops or y_jumps or img_changes:
            lines.append(current_line)
            current_line = [a]
        else:
            current_line.append(a)
    lines.append(current_line)
    return lines


def build_interpolation(lines):
    """Build line info with averaged y-coordinates for interpolation."""
    line_info = []
    for line in lines:
        times = [p["timestamp_ms"] for p in line]
        xs = [p["x_ratio"] for p in line]
        ys = [p["y_ratio"] for p in line]
        avg_y = sum(ys) / len(ys)
        line_info.append({
            "start_ms": times[0],
            "end_ms": times[-1],
            "image_index": line[0]["image_index"],
            "avg_y": avg_y,
            "times": times,
            "xs": xs,
        })
    return line_info


def get_position_at_time(t_ms, line_info):
    """Get interpolated (image_index, x_ratio, y_ratio) at a given time.

    Each line owns [its start, next line's start). After the last annotation
    point, x is extrapolated at the last segment's speed until the next line.
    """
    n = len(line_info)

    for i, li in enumerate(line_info):
        t_start = li["start_ms"]
        t_end = line_info[i + 1]["start_ms"] if i + 1 < n else float("inf")

        if t_start <= t_ms < t_end:
            if len(li["times"]) == 1:
                return li["image_index"], li["xs"][0], li["avg_y"]

            if t_ms <= li["end_ms"]:
                f = interp1d(li["times"], li["xs"], kind="linear")
                return li["image_index"], float(f(t_ms)), li["avg_y"]
            else:
                dt_last = li["times"][-1] - li["times"][-2]
                dx_last = li["xs"][-1] - li["xs"][-2]
                speed = dx_last / dt_last if dt_last > 0 else 0
                x = li["xs"][-1] + speed * (t_ms - li["end_ms"])
                x = max(0.0, min(x, 1.0))
                return li["image_index"], x, li["avg_y"]

    li = line_info[0]
    return li["image_index"], li["xs"][0], li["avg_y"]


class ScoreFollowingDataset(Dataset):
    """Dataset for score-following training.

    Each sample provides:
      - The sheet music image for the current position
      - Audio segment starting at the given timestamp
      - Start position (x, y) on the image
      - Target: next N positions (x, y) — positions on a different image
        are flagged with image_index so the model can learn page transitions
    """

    def __init__(self, data_dirs, config, processor=None):
        self.config = config
        self.processor = processor
        self.samples = []

        for data_dir in data_dirs:
            self._load_piece(Path(data_dir))

    def _load_piece(self, data_dir):
        ann_files = list(data_dir.glob("annotations_*.json"))
        if not ann_files:
            return

        with open(ann_files[0]) as f:
            annotations = json.load(f)

        audio_path = data_dir / annotations["audio_filename"]
        image_paths = [data_dir / f for f in annotations["image_filenames"]]
        audio_duration_ms = annotations["audio_duration_ms"]

        lines = detect_lines(annotations)
        line_info = build_interpolation(lines)

        # Load audio
        try:
            import soundfile as sf
            audio_data, sr = sf.read(str(audio_path))
        except Exception:
            import librosa
            audio_data, sr = librosa.load(str(audio_path), sr=self.config.audio_sample_rate, mono=True)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to target sample rate if needed
        if sr != self.config.audio_sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.config.audio_sample_rate)

        # Generate samples at various start positions
        # Use every annotation point and also random points between them
        min_t = line_info[0]["start_ms"]
        max_t = line_info[-1]["end_ms"]
        pred_window = self.config.pred_steps * self.config.pred_interval_ms

        # Sample start times: every sample_shift_ms where we have enough future data
        start_times = list(range(int(min_t), int(max_t - pred_window), self.config.sample_shift_ms))

        for t_start in start_times:
            # Check that audio covers this segment
            audio_end_ms = t_start + self.config.audio_segment_sec * 1000
            if audio_end_ms > audio_duration_ms:
                continue

            # Get start position
            img_idx_start, x_start, y_start = get_position_at_time(t_start, line_info)

            # Get target positions
            targets = []
            valid = True
            for step in range(1, self.config.pred_steps + 1):
                t_target = t_start + step * self.config.pred_interval_ms
                if t_target > max_t:
                    valid = False
                    break
                img_idx_t, x_t, y_t = get_position_at_time(t_target, line_info)
                targets.append((x_t, y_t, img_idx_t))

            if not valid or len(targets) < self.config.pred_steps:
                continue

            self.samples.append({
                "audio_data": audio_data,
                "audio_sr": self.config.audio_sample_rate,
                "t_start_ms": t_start,
                "image_paths": image_paths,
                "image_index": img_idx_start,
                "start_pos": (x_start, y_start, img_idx_start),
                "targets": targets,
                "line_info": line_info,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cfg = self.config

        # Extract audio segment
        t_start_sec = sample["t_start_ms"] / 1000.0
        sr = sample["audio_sr"]
        start_sample = int(t_start_sec * sr)
        end_sample = start_sample + int(cfg.audio_segment_sec * sr)
        audio_segment = sample["audio_data"][start_sample:end_sample]

        # Pad if too short
        expected_len = int(cfg.audio_segment_sec * sr)
        if len(audio_segment) < expected_len:
            audio_segment = np.pad(audio_segment, (0, expected_len - len(audio_segment)))

        # Load the current image
        from PIL import Image
        img_path = sample["image_paths"][sample["image_index"]]
        image = Image.open(str(img_path)).convert("RGB")

        # Target coordinates: (pred_steps, 2) -> (x, y)
        target_xy = torch.tensor(
            [(t[0], t[1]) for t in sample["targets"]], dtype=torch.float32,
        )
        # Target image indices: (pred_steps,) -> class labels for cross-entropy
        target_img = torch.tensor(
            [t[2] for t in sample["targets"]], dtype=torch.long,
        )

        # Start position (x, y, image_number)
        start_pos = torch.tensor(
            [sample["start_pos"][0], sample["start_pos"][1]], dtype=torch.float32,
        )
        start_img = torch.tensor(sample["start_pos"][2], dtype=torch.long)

        return {
            "audio": torch.tensor(audio_segment, dtype=torch.float32),
            "image": image,
            "start_pos": start_pos,
            "start_img": start_img,
            "target_xy": target_xy,
            "target_img": target_img,
        }


def collate_fn(batch):
    """Custom collate that handles PIL images."""
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "images": [b["image"] for b in batch],
        "start_pos": torch.stack([b["start_pos"] for b in batch]),
        "start_img": torch.stack([b["start_img"] for b in batch]),
        "target_xy": torch.stack([b["target_xy"] for b in batch]),
        "target_img": torch.stack([b["target_img"] for b in batch]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class PositionPredictionHead(nn.Module):
    """Regression head that predicts future (x, y) + image class logits."""

    def __init__(self, hidden_size, pred_steps, max_num_images, start_pos_dim=2):
        super().__init__()
        self.pred_steps = pred_steps
        self.max_num_images = max_num_images
        self.out_per_step = 2 + max_num_images  # (x, y) + image logits
        # Fuse hidden representation with start position (x, y) + image embedding
        self.img_embed = nn.Embedding(max_num_images, 16)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size + start_pos_dim + 16, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, pred_steps * self.out_per_step),
        )

    def forward(self, hidden_states, start_pos, start_img):
        """
        hidden_states: (B, hidden_size) - pooled representation from Phi-4
        start_pos: (B, 2) - starting (x, y)
        start_img: (B,) - starting image index (long)

        Returns:
            xy: (B, pred_steps, 2) - sigmoid-activated coordinates
            img_logits: (B, pred_steps, max_num_images) - raw logits for cross-entropy
        """
        img_emb = self.img_embed(start_img)  # (B, 16)
        combined = torch.cat([hidden_states, start_pos, img_emb], dim=-1)
        out = self.proj(combined)
        out = out.view(-1, self.pred_steps, self.out_per_step)
        xy = torch.sigmoid(out[:, :, :2])
        img_logits = out[:, :, 2:]
        return xy, img_logits


class ScoreFollowingModel(nn.Module):
    """
    Phi-4-multimodal-instruct fine-tuned for score following.

    Uses the model's native multimodal capabilities to process both
    the sheet music image and audio, then predicts future positions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import AutoModelForCausalLM, AutoProcessor
        import flash_attn  # noqa: F401 – check availability

        # Load Phi-4-multimodal
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        hidden_size = self.backbone.config.hidden_size

        # Freeze backbone initially (we'll use LoRA or selective unfreezing)
        if config.use_lora:
            self._apply_lora(config)
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Prediction head (always trainable)
        self.pred_head = PositionPredictionHead(
            hidden_size=hidden_size,
            pred_steps=config.pred_steps,
            max_num_images=config.max_num_images,
        )

    def _apply_lora(self, config):
        """Apply LoRA adapters to the backbone for efficient fine-tuning."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, audio, images, start_pos, start_img):
        """
        audio: (B, num_samples) - raw audio waveform at 16kHz
        images: list of B PIL Images
        start_pos: (B, 2) - start (x, y) coordinates
        start_img: (B,) - start image index (long)

        Returns:
            xy: (B, pred_steps, 2) - predicted coordinates
            img_logits: (B, pred_steps, max_num_images) - image class logits
        """
        batch_size = len(images)
        device = start_pos.device

        # Build prompts for Phi-4 multimodal
        prompts = []
        all_images = []
        all_audios = []
        for i in range(batch_size):
            x, y = start_pos[i].tolist()
            img_num = start_img[i].item()
            prompt = (
                f"<|image_1|><|audio_1|>"
                f"Current position on the music score: x={x:.4f}, y={y:.4f}, image={img_num}. "
                f"Predict the next positions."
            )
            prompts.append(prompt)
            all_images.append(images[i])
            # Phi-4 audio: (num_samples,) numpy array at 16kHz
            all_audios.append(audio[i].cpu().numpy())

        # Process inputs through Phi-4 processor
        inputs = self.processor(
            text=prompts,
            images=all_images,
            audios=all_audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward through backbone - get hidden states
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Pool the last hidden state (use the last token's representation)
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
        # Use mean pooling over sequence for a robust representation
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = last_hidden.mean(dim=1)

        pooled = pooled.float()  # Cast from bf16 to float32 for the head

        # Predict positions
        return self.pred_head(pooled, start_pos, start_img)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def score_following_loss(pred_xy, img_logits, target_xy, target_img):
    """
    pred_xy:     (B, pred_steps, 2) - predicted coordinates
    img_logits:  (B, pred_steps, max_num_images) - image class logits
    target_xy:   (B, pred_steps, 2) - ground truth coordinates
    target_img:  (B, pred_steps) - ground truth image indices (long)
    """
    coord_loss = F.mse_loss(pred_xy, target_xy)
    # Reshape for cross_entropy: (B*pred_steps, max_num_images) vs (B*pred_steps,)
    img_loss = F.cross_entropy(
        img_logits.reshape(-1, img_logits.size(-1)),
        target_img.reshape(-1),
    )
    return coord_loss + img_loss


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(config=None):
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = ScoreFollowingDataset(config.data_dirs, config)
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        raise RuntimeError("No training samples generated. Check data directories.")

    # Split into train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = ScoreFollowingModel(config).to(device)

    # Optimizer: different LR for backbone vs head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.pred_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.learning_rate * 0.1},
        {"params": head_params, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.learning_rate * 0.1, config.learning_rate],
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="cos",
    )

    # Training
    os.makedirs(config.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            start_pos = batch["start_pos"].to(device)
            start_img = batch["start_img"].to(device)
            target_xy = batch["target_xy"].to(device)
            target_img = batch["target_img"].to(device)
            images = batch["images"]  # List of PIL Images

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_xy, img_logits = model(audio, images, start_pos, start_img)
                loss = score_following_loss(pred_xy, img_logits, target_xy, target_img)
                loss = loss / config.grad_accum_steps

            loss.backward()
            epoch_loss += loss.item() * config.grad_accum_steps

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                start_pos = batch["start_pos"].to(device)
                start_img = batch["start_img"].to(device)
                target_xy = batch["target_xy"].to(device)
                target_img = batch["target_img"].to(device)
                images = batch["images"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_xy, img_logits = model(audio, images, start_pos, start_img)
                    loss = score_following_loss(pred_xy, img_logits, target_xy, target_img)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config.num_epochs}  "
              f"train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config),
            }, save_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
            }, save_path)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Export for mobile deployment (ONNX)
# ──────────────────────────────────────────────────────────────────────────────

def export_for_mobile(checkpoint_path, output_path="model_mobile.onnx"):
    """Export the trained model to ONNX format for mobile deployment.

    For phone/tablet deployment, use ONNX Runtime Mobile or convert further
    to CoreML (iOS) or TFLite (Android).
    """
    config = Config()
    model = ScoreFollowingModel(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # For mobile export, we extract just the prediction head and
    # a distilled audio encoder. The full Phi-4 model is too large
    # for mobile — see export_distilled_for_mobile() below.
    print("NOTE: Full Phi-4 model is too large for direct mobile deployment.")
    print("Use export_distilled_for_mobile() after knowledge distillation.")
    print(f"Full model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")


def export_distilled_for_mobile(checkpoint_path, output_dir="mobile_model"):
    """Export a distilled/student model for mobile deployment.

    Strategy for mobile deployment:
    1. Train the full Phi-4 model (train.py)
    2. Use knowledge distillation to train a smaller student model
    3. Export the student model to ONNX
    4. Convert to CoreML (iOS) or TFLite (Android) using respective tools

    The student model uses:
    - Whisper-tiny for audio encoding (39M params)
    - MobileNetV3-small for image encoding (2.5M params)
    - Lightweight prediction head
    Total: ~45M params — suitable for real-time mobile inference
    """
    os.makedirs(output_dir, exist_ok=True)

    # Student model architecture (same as in train_distilled)
    from transformers import WhisperModel
    from torchvision.models import mobilenet_v3_small

    config = Config()

    class MobileScoreFollower(nn.Module):
        def __init__(self, pred_steps, max_num_images):
            super().__init__()
            self.pred_steps = pred_steps
            self.max_num_images = max_num_images
            self.out_per_step = 2 + max_num_images

            whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
            self.audio_encoder = whisper.encoder
            self.audio_proj = nn.Linear(384, 128)

            mobilenet = mobilenet_v3_small(pretrained=True)
            self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
            self.image_proj = nn.Linear(576, 128)

            self.img_embed = nn.Embedding(max_num_images, 16)
            # 128 (audio) + 128 (image) + 2 (xy) + 16 (img embed) = 274
            self.predictor = nn.Sequential(
                nn.Linear(274, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, pred_steps * self.out_per_step),
            )

        def forward(self, audio_features, image, start_pos, start_img):
            audio_out = self.audio_encoder(audio_features).last_hidden_state
            audio_emb = self.audio_proj(audio_out.mean(dim=1))
            img_emb = self.image_encoder(image).squeeze(-1).squeeze(-1)
            img_emb = self.image_proj(img_emb)
            img_start_emb = self.img_embed(start_img)
            combined = torch.cat([audio_emb, img_emb, start_pos, img_start_emb], dim=-1)
            out = self.predictor(combined)
            out = out.view(-1, self.pred_steps, self.out_per_step)
            xy = torch.sigmoid(out[:, :, :2])
            img_logits = out[:, :, 2:]
            return xy, img_logits

    student = MobileScoreFollower(config.pred_steps, config.max_num_images)
    print(f"Student model size: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params")

    # Export to ONNX
    student.eval()
    dummy_audio = torch.randn(1, 80, 3000)
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_pos = torch.tensor([[0.5, 0.5]])
    dummy_img = torch.tensor([0], dtype=torch.long)

    onnx_path = os.path.join(output_dir, "score_follower.onnx")
    torch.onnx.export(
        student,
        (dummy_audio, dummy_image, dummy_pos, dummy_img),
        onnx_path,
        input_names=["audio_features", "image", "start_pos", "start_img"],
        output_names=["pred_xy", "img_logits"],
        dynamic_axes={
            "audio_features": {0: "batch", 2: "time"},
            "image": {0: "batch"},
            "start_pos": {0: "batch"},
            "start_img": {0: "batch"},
            "pred_xy": {0: "batch"},
            "img_logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved to: {onnx_path}")
    print("\nNext steps for mobile deployment:")
    print("  iOS:     python -m coremltools.converters.onnx score_follower.onnx")
    print("  Android: python -m onnxruntime.tools.convert_onnx_models_to_ort score_follower.onnx")


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge distillation training
# ──────────────────────────────────────────────────────────────────────────────

def train_distilled(teacher_checkpoint, config=None):
    """Train a mobile-sized student model using knowledge distillation
    from the fine-tuned Phi-4 teacher model.
    """
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher
    teacher = ScoreFollowingModel(config).to(device)
    ckpt = torch.load(teacher_checkpoint, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student (defined inline for simplicity; same as in export)
    from transformers import WhisperModel, WhisperFeatureExtractor
    from torchvision.models import mobilenet_v3_small
    from torchvision import transforms

    class MobileScoreFollower(nn.Module):
        def __init__(self, pred_steps, max_num_images):
            super().__init__()
            self.pred_steps = pred_steps
            self.max_num_images = max_num_images
            self.out_per_step = 2 + max_num_images
            whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
            self.audio_encoder = whisper.encoder
            self.audio_proj = nn.Linear(384, 128)
            mobilenet = mobilenet_v3_small(pretrained=True)
            self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
            self.image_proj = nn.Linear(576, 128)
            self.img_embed = nn.Embedding(max_num_images, 16)
            # 128 (audio) + 128 (image) + 2 (xy) + 16 (img embed) = 274
            self.predictor = nn.Sequential(
                nn.Linear(274, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, pred_steps * self.out_per_step),
            )

        def forward(self, audio_features, image, start_pos, start_img):
            audio_out = self.audio_encoder(audio_features).last_hidden_state
            audio_emb = self.audio_proj(audio_out.mean(dim=1))
            img_emb = self.image_encoder(image).squeeze(-1).squeeze(-1)
            img_emb = self.image_proj(img_emb)
            img_start_emb = self.img_embed(start_img)
            combined = torch.cat([audio_emb, img_emb, start_pos, img_start_emb], dim=-1)
            out = self.predictor(combined)
            out = out.view(-1, self.pred_steps, self.out_per_step)
            xy = torch.sigmoid(out[:, :, :2])
            img_logits = out[:, :, 2:]
            return xy, img_logits

    student = MobileScoreFollower(config.pred_steps, config.max_num_images).to(device)
    whisper_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ScoreFollowingDataset(config.data_dirs, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

    distill_temp = 2.0
    alpha_distill = 0.7  # weight for distillation loss vs ground truth

    for epoch in range(config.num_epochs):
        student.train()
        total_loss = 0.0

        for batch in loader:
            audio = batch["audio"].to(device)
            start_pos = batch["start_pos"].to(device)
            start_img = batch["start_img"].to(device)
            target_xy = batch["target_xy"].to(device)
            target_img = batch["target_img"].to(device)
            images = batch["images"]

            # Teacher predictions
            with torch.no_grad():
                teacher_xy, teacher_img_logits = teacher(audio, images, start_pos, start_img)

            # Prepare student inputs
            # Audio -> mel features for Whisper
            mel_features = []
            for a in audio:
                feat = whisper_extractor(a.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                mel_features.append(feat.input_features.squeeze(0))
            mel_features = torch.stack(mel_features).to(device)

            # Images -> transformed tensors
            img_tensors = torch.stack([img_transform(img) for img in images]).to(device)

            # Student predictions
            student_xy, student_img_logits = student(mel_features, img_tensors, start_pos, start_img)

            # Combined loss: distillation + ground truth
            distill_loss = (
                F.mse_loss(student_xy, teacher_xy)
                + F.mse_loss(student_img_logits, teacher_img_logits)
            )
            gt_loss = score_following_loss(student_xy, student_img_logits, target_xy, target_img)
            loss = alpha_distill * distill_loss + (1 - alpha_distill) * gt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Distill Epoch {epoch+1}/{config.num_epochs}  loss={total_loss/len(loader):.6f}")

    # Save student
    save_path = os.path.join(config.output_dir, "student_model.pt")
    torch.save(student.state_dict(), save_path)
    print(f"Student model saved to: {save_path}")
    return student


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score following model training")
    parser.add_argument("--mode", choices=["train", "distill", "export"], default="train")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for distill/export")
    parser.add_argument("--data-dirs", nargs="+", default=["data/Hands_Across_the_Sea"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pred-steps", type=int, default=10, help="Number of future positions to predict")
    parser.add_argument("--pred-interval-ms", type=int, default=500, help="Interval between predicted positions (ms)")
    parser.add_argument("--sample-shift-ms", type=int, default=250, help="Shift between training samples (ms)")

    args = parser.parse_args()

    config = Config()
    config.data_dirs = args.data_dirs
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.pred_steps = args.pred_steps
    config.pred_interval_ms = args.pred_interval_ms
    config.sample_shift_ms = args.sample_shift_ms

    if args.mode == "train":
        train(config)
    elif args.mode == "distill":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for distillation")
        train_distilled(args.checkpoint, config)
    elif args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for export")
        export_distilled_for_mobile(args.checkpoint)
