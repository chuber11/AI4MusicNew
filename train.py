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
from tqdm import tqdm

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
    dev_dirs   = ["data/dev/*"]   # if empty, 10% of train data is held out automatically

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

        import glob as _glob
        expanded = []
        for pattern in data_dirs:
            matches = _glob.glob(str(pattern))
            expanded.extend(matches if matches else [pattern])

        for data_dir in expanded:
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

        min_t = line_info[0]["start_ms"]
        max_t = line_info[-1]["end_ms"]
        audio_len_ms = self.config.audio_length_sec * 1000
        shift_ms = int(self.config.sample_shift_sec * 1000)

        start_times = list(range(int(min_t), int(max_t - audio_len_ms), shift_ms))

        # Dense target resolution: 200 uniformly spaced points across the audio segment
        _N_DENSE = 200

        for t_start in start_times:
            # Check that audio covers this segment
            if t_start + audio_len_ms > audio_duration_ms:
                continue

            # Get start position
            img_idx_start, x_start, y_start = get_position_at_time(t_start, line_info)

            # Get targets: 200 uniformly spaced points across the audio segment
            targets = []
            for i in range(_N_DENSE):
                t_target = t_start + i * audio_len_ms / (_N_DENSE - 1)
                img_idx_t, x_t, y_t = get_position_at_time(t_target, line_info)
                targets.append((x_t, y_t, img_idx_t))

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
        end_sample = start_sample + int(cfg.audio_length_sec * sr)
        audio_segment = sample["audio_data"][start_sample:end_sample]

        # Pad if too short
        expected_len = int(cfg.audio_length_sec * sr)
        if len(audio_segment) < expected_len:
            audio_segment = np.pad(audio_segment, (0, expected_len - len(audio_segment)))

        # Load all pages so Phi-4 has full visual context
        from PIL import Image
        all_images = [Image.open(str(p)).convert("RGB") for p in sample["image_paths"]]

        # Dense targets: (200, 2) uniformly across the audio segment
        target_xy = torch.tensor(
            [(t[0], t[1]) for t in sample["targets"]], dtype=torch.float32,
        )
        # Target image indices: (200,) -> class labels for cross-entropy
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
            "all_images": all_images,       # list of PIL Images (one per page)
            "start_pos": start_pos,
            "start_img": start_img,
            "target_xy": target_xy,
            "target_img": target_img,
        }


def collate_fn(batch, processor, audio_sample_rate):
    """Run the processor on the whole batch inside the DataLoader worker.

    Each worker has a forked copy of the processor, so preprocessing is
    parallelised across CPUs while the GPU runs the previous batch.
    """
    # Build per-sample prompts (num_pages may differ between pieces)
    prompts = [
        "".join(f"<|image_{j+1}|>" for j in range(len(b["all_images"])))
        + "<|pos|><|page|><|audio_1|>"
        for b in batch
    ]
    flat_images = [img for b in batch for img in b["all_images"]]
    audios      = [(b["audio"].numpy(), audio_sample_rate) for b in batch]

    inputs = processor(
        text=prompts,
        images=flat_images,
        audios=audios,
        return_tensors="pt",
        padding=True,
    )

    inputs["start_pos"] = torch.stack([b["start_pos"] for b in batch])
    inputs["start_img"] = torch.stack([b["start_img"] for b in batch])
    inputs["target_xy"] = torch.stack([b["target_xy"] for b in batch])
    inputs["target_img"] = torch.stack([b["target_img"] for b in batch])
    return inputs


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def fourier_encode(x, num_freqs=8):
    """Encode scalar(s) in [0,1] with sinusoidal Fourier features.

    Args:
        x: (..., D) tensor of values in [0, 1]
        num_freqs: number of frequency bands (exponentially spaced)

    Returns:
        (..., D * 2 * num_freqs) — sin and cos at each frequency
    """
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    x_freq = x.unsqueeze(-1) * freqs * math.pi          # (..., D, F)
    encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
    return encoded.flatten(-2)                           # (..., D*2*F)


class ScoreFollowingModel(nn.Module):
    """
    Phi-4-multimodal-instruct fine-tuned for score following.

    Input sequence layout (audio last for KV-cache reuse at inference):
        <|image_1|> ... <|image_N|>  <|pos|>  <|page|>  <|audio_1|>

    - Vision-LoRA: frozen (used as-is for image tokens)
    - Speech-LoRA: fine-tuned (handles all non-image tokens: pos, page, audio)
    - pos/page tokens: Fourier-encoded (x,y) and learned page embedding,
      projected to hidden_size and injected directly into the transformer
      sequence so every attention layer can attend to them.
    - Head: single Linear layer — the transformer already does the fusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import AutoModelForCausalLM, AutoProcessor

        # ── Processor: add placeholder tokens for position and page ──────────
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=True,
        )
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|pos|>", "<|page|>"]}
        )
        self.pos_token_id  = self.processor.tokenizer.convert_tokens_to_ids("<|pos|>")
        self.page_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|page|>")

        # ── Backbone: load with native vision-lora and speech-lora ───────────
        # Newer PEFT (≥0.13) requires prepare_inputs_for_generation on the inner
        # Phi4MMModel when Phi-4's __init__ calls get_peft_model(self.model, ...).
        # Phi4MMModel (the decoder) doesn't have it — only the CausalLM wrapper does.
        # We patch PEFT once before loading so it doesn't raise; we never call
        # generate(), so the no-op lambda is never invoked.
        try:
            import peft.peft_model as _peft_model
            _orig = _peft_model.PeftModelForCausalLM.__init__
            def _patched(self, model, peft_config, adapter_name="default", **kw):
                if not hasattr(model, "prepare_inputs_for_generation"):
                    model.prepare_inputs_for_generation = lambda *a, **k: {}
                _orig(self, model, peft_config, adapter_name=adapter_name, **kw)
            _peft_model.PeftModelForCausalLM.__init__ = _patched
        except Exception:
            pass

        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        # Properly initialise _gradient_checkpointing_func on all submodules
        # (Phi-4's cached code sets the gradient_checkpointing flag but never
        # calls enable(), so _gradient_checkpointing_func is missing at runtime)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Extend embedding table for the two new tokens
        self.backbone.resize_token_embeddings(len(self.processor.tokenizer))

        hidden_size = self.backbone.config.hidden_size  # 3072
        self._hidden_size = hidden_size

        # ── Freeze everything; unfreeze speech-LoRA only ─────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Auto-detect LoRA adapter names from PEFT parameter names.
        # PEFT stores weights as  ...lora_A.{adapter_name}.weight
        lora_adapters = set()
        for name, _ in self.backbone.named_parameters():
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ("lora_A", "lora_B") and i + 1 < len(parts):
                    lora_adapters.add(parts[i + 1])

        # Train every adapter except the vision one (kept frozen as-is)
        speech_adapters = lora_adapters - {"vision"}
        print(f"All LoRA adapters found : {sorted(lora_adapters)}")
        print(f"Adapters to fine-tune   : {sorted(speech_adapters)}")

        n_speech = 0
        for name, param in self.backbone.named_parameters():
            if any(f".{a}." in name for a in speech_adapters):
                param.requires_grad = True
                n_speech += param.numel()
        print(f"Trainable speech-LoRA parameters: {n_speech:,}")

        # ── Position and page token projections (always trainable) ───────────
        _device = next(self.backbone.parameters()).device
        pos_enc_dim = 2 * 2 * config.pos_num_freqs      # 32 for default 8 freqs
        self.pos_proj  = nn.Linear(pos_enc_dim, hidden_size, bias=False).to(_device)
        self.page_proj = nn.Embedding(config.max_num_images, hidden_size).to(_device)

        # ── Single linear head — one position prediction per token ───────────
        self.head = nn.Linear(hidden_size, 2 + config.max_num_images).to(_device)

    def forward(self, inputs, start_pos, start_img):
        """
        inputs:     dict   processor output (input_ids, pixel_values, etc.) on device
        start_pos:  (B, 2) normalized (x, y) ∈ [0, 1]
        start_img:  (B,)   long, current page index

        Returns:
            xy:         (B, seq_len, 2)              sigmoid coordinates at each token
            img_logits: (B, seq_len, max_num_images) raw page logits at each token
        """
        # ── Compute pos/page token embeddings ─────────────────────────────────
        pos_enc  = fourier_encode(start_pos, self.config.pos_num_freqs)     # (B, 32)
        pos_emb  = self.pos_proj(pos_enc.float()).to(torch.bfloat16)        # (B, H)
        page_emb = self.page_proj(start_img).to(torch.bfloat16)            # (B, H)

        # Find the sequence positions of our placeholder tokens
        input_ids  = inputs["input_ids"]
        pos_locs   = (input_ids == self.pos_token_id ).nonzero(as_tuple=False)
        page_locs  = (input_ids == self.page_token_id).nonzero(as_tuple=False)

        # ── Hook: inject pos/page embeddings after Phi-4's image/audio ────────
        # By the time the first transformer layer runs, Phi-4 has already replaced
        # image and audio placeholder tokens with encoder outputs.  We replace our
        # two placeholder positions with the Fourier / page embeddings here.
        injected = [False]

        def _inject(module, args, kwargs):
            if injected[0]:
                return args, kwargs
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return args, kwargs
            # Clone before writing — in-place modification breaks gradient checkpointing
            # (checkpointing saves the tensor at version N; inplace bumps it to N+1)
            hidden = hidden.clone()
            for row in pos_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = pos_emb[b]
            for row in page_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = page_emb[b]
            injected[0] = True
            if args:
                return (hidden,) + args[1:], kwargs
            else:
                return args, {**kwargs, "hidden_states": hidden}

        handle = self.backbone.model.layers[0].register_forward_pre_hook(
            _inject, with_kwargs=True,
        )
        try:
            outputs = self.backbone(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            handle.remove()

        # ── Apply head at every sequence position ─────────────────────────────
        last_hidden = outputs.hidden_states[-1].float()   # (B, seq_len, H)
        out        = self.head(last_hidden)               # (B, seq_len, 2 + max_pages)
        xy         = torch.sigmoid(out[:, :, :2])         # (B, seq_len, 2)
        img_logits = out[:, :, 2:]                        # (B, seq_len, max_pages)
        return xy, img_logits


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def score_following_loss(pred_xy, img_logits, target_xy, target_img):
    """Per-token loss: interpolate dense targets to match actual sequence length.

    pred_xy:     (B, seq_len, 2)              predicted coordinates at every token
    img_logits:  (B, seq_len, max_num_images) predicted page logits at every token
    target_xy:   (B, num_dense, 2)            ground truth at num_dense uniform times
    target_img:  (B, num_dense)               ground truth page indices (long)
    """
    B, seq_len, _ = pred_xy.shape
    num_dense = target_xy.shape[1]

    if num_dense != seq_len:
        # Interpolate xy targets (linear) and img targets (nearest) to seq_len
        target_xy = F.interpolate(
            target_xy.permute(0, 2, 1),   # (B, 2, num_dense)
            size=seq_len,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)                # (B, seq_len, 2)

        target_img = F.interpolate(
            target_img.float().unsqueeze(1),  # (B, 1, num_dense)
            size=seq_len,
            mode="nearest",
        ).squeeze(1).long()               # (B, seq_len)

    coord_loss = F.mse_loss(pred_xy, target_xy)
    img_loss   = F.cross_entropy(
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

    # Dataset (annotation loading only — no processor calls yet)
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

    # Model — must be created before DataLoaders so we can pass its processor
    model = ScoreFollowingModel(config)

    from functools import partial
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

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps

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

    print("Training starts now.")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            start_pos  = batch.pop("start_pos").to(device)
            start_img  = batch.pop("start_img").to(device)
            target_xy  = batch.pop("target_xy").to(device)
            target_img = batch.pop("target_img").to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_xy, img_logits = model(inputs, start_pos, start_img)
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
            for batch in tqdm(val_loader):
                start_pos  = batch.pop("start_pos").to(device)
                start_img  = batch.pop("start_img").to(device)
                target_xy  = batch.pop("target_xy").to(device)
                target_img = batch.pop("target_img").to(device)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_xy, img_logits = model(inputs, start_pos, start_img)
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
# ONNX export — single file, LLM weights stored once
# ──────────────────────────────────────────────────────────────────────────────

class _StreamingScoreFollower(nn.Module):
    """Single ONNX graph handling both prefix and audio-decode passes.

    The transformer (LLM) weights are stored exactly once in this single file.
    The caller pre-computes inputs_embeds outside the ONNX graph using
    compute_prefix_embeds() or compute_audio_embeds().

    Prefix pass  — kv_len = 0 (empty cache):
        inputs_embeds = image tokens + pos token + page token
        past_keys / past_values have shape (L, B, heads, 0, head_dim)
        → builds and returns the prefix KV cache

    Audio pass   — kv_len = prefix_len (cached prefix):
        inputs_embeds = audio tokens for this chunk
        past_keys / past_values = prefix KV cache from prefix pass
        → returns predictions; caller discards new_past_keys/values and
          reuses the same fixed prefix cache for the next audio chunk
    """

    def __init__(self, model: ScoreFollowingModel):
        super().__init__()
        self.transformer = model.backbone.model  # decoder stack, stored once
        self.head        = model.head

    def forward(
        self,
        inputs_embeds:  torch.Tensor,   # (B, seq_len, H)
        attention_mask: torch.Tensor,   # (B, kv_len + seq_len)
        past_keys:      torch.Tensor,   # (L, B, heads, kv_len, head_dim)
        past_values:    torch.Tensor,   # (L, B, heads, kv_len, head_dim)
    ):
        L   = past_keys.shape[0]
        pkv = tuple((past_keys[i], past_values[i]) for i in range(L))

        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )

        # Re-stack KV cache as tensors (ONNX needs concrete tensors, not tuples)
        new_past_keys   = torch.stack([kv[0] for kv in out.past_key_values])
        new_past_values = torch.stack([kv[1] for kv in out.past_key_values])

        # Apply head at every token position — no pooling needed
        result     = self.head(out.last_hidden_state.float())  # (B, seq_len, 2 + max_pages)
        pred_xy    = torch.sigmoid(result[:, :, :2])           # (B, seq_len, 2)
        img_logits = result[:, :, 2:]                          # (B, seq_len, max_pages)

        return pred_xy, img_logits, new_past_keys, new_past_values


def _capture_inputs_embeds(model: ScoreFollowingModel, backbone_inputs: dict) -> torch.Tensor:
    """Run the backbone up to (but not including) layer 0, capturing inputs_embeds.

    Hooks into backbone.model.layers[0] to intercept the merged hidden states
    (image + audio + text token embeddings already merged) just before the
    first transformer layer processes them.
    """
    captured = {}

    def _hook(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is not None and "embeds" not in captured:
            captured["embeds"] = hidden.detach().clone()

    handle = model.backbone.model.layers[0].register_forward_pre_hook(
        _hook, with_kwargs=True,
    )
    with torch.no_grad():
        model.backbone(**backbone_inputs, use_cache=False, return_dict=True)
    handle.remove()

    return captured["embeds"]   # (B, seq_len, H)


def compute_prefix_embeds(
    model:     ScoreFollowingModel,
    images:    list,            # B lists of PIL Images, each list = one sample's pages
    start_pos: torch.Tensor,    # (B, 2)  normalized (x, y)
    start_img: torch.Tensor,    # (B,)    long, current page index
) -> torch.Tensor:
    """Compute merged inputs_embeds for the prefix (images + pos + page tokens).

    Not exported to ONNX — call this in Python once per piece / start position.
    The result is passed as inputs_embeds to the ONNX graph for the prefix pass.
    """
    cfg       = model.config
    B         = start_pos.shape[0]
    device    = start_pos.device
    num_pages = len(images[0])

    img_tags = "".join(f"<|image_{j+1}|>" for j in range(num_pages))
    texts    = [f"{img_tags}<|pos|><|page|>"] * B

    inputs = model.processor(
        text=texts, images=images, return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Prepare pos/page embeddings for injection
    pos_enc  = fourier_encode(start_pos, cfg.pos_num_freqs)
    pos_emb  = model.pos_proj(pos_enc.float()).to(torch.bfloat16)
    page_emb = model.page_proj(start_img).to(torch.bfloat16)

    input_ids = inputs["input_ids"]
    pos_locs  = (input_ids == model.pos_token_id ).nonzero(as_tuple=False)
    page_locs = (input_ids == model.page_token_id).nonzero(as_tuple=False)
    injected  = [False]

    captured = {}

    def _inject_and_capture(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            return args, kwargs
        hidden = hidden.clone()
        if not injected[0]:
            for row in pos_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = pos_emb[b]
            for row in page_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = page_emb[b]
            injected[0] = True
        if "embeds" not in captured:
            captured["embeds"] = hidden.detach().clone()
        if args:
            return (hidden,) + args[1:], kwargs
        else:
            return args, {**kwargs, "hidden_states": hidden}

    handle = model.backbone.model.layers[0].register_forward_pre_hook(
        _inject_and_capture, with_kwargs=True,
    )
    with torch.no_grad():
        model.backbone(**inputs, use_cache=False, return_dict=True)
    handle.remove()

    return captured["embeds"]   # (B, prefix_seq_len, H)


def compute_audio_embeds(
    model: ScoreFollowingModel,
    audio: torch.Tensor,    # (B, T)  raw 16 kHz waveform
) -> torch.Tensor:
    """Compute inputs_embeds for a single audio chunk.

    Not exported to ONNX — call this in Python for each new audio chunk.
    The result is passed as inputs_embeds to the ONNX graph for the audio pass.
    """
    B      = audio.shape[0]
    device = audio.device

    audio_inputs = model.processor(
        text=["<|audio_1|>"] * B,
        audios=[(audio[i].cpu().numpy(), model.config.audio_sample_rate) for i in range(B)],
        return_tensors="pt",
        padding=True,
    )
    audio_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in audio_inputs.items()}

    return _capture_inputs_embeds(model, audio_inputs)   # (B, audio_seq_len, H)


def export_onnx(checkpoint_path, output_dir="onnx_export"):
    """Export as a single ONNX graph — LLM weights stored exactly once.

    File: score_follower.onnx
        Inputs:
            inputs_embeds   (B, seq_len, H)                 pre-computed embeddings
            attention_mask  (B, kv_len + seq_len)
            past_keys       (L, B, heads, kv_len, head_dim)
            past_values     (L, B, heads, kv_len, head_dim)
        Outputs:
            pred_xy         (B, seq_len, 2)
            img_logits      (B, seq_len, max_pages)
            new_past_keys   (L, B, heads, kv_len + seq_len, head_dim)
            new_past_values (L, B, heads, kv_len + seq_len, head_dim)

    Inference flow:
        # ── run once per piece / per new start position ───────────────────────
        prefix_embeds = compute_prefix_embeds(model, images, start_pos, start_img)
        prefix_len    = prefix_embeds.shape[1]
        prefix_mask   = torch.ones(B, prefix_len, dtype=torch.long)
        empty_keys    = torch.zeros(L, B, heads, 0, head_dim)
        empty_values  = torch.zeros(L, B, heads, 0, head_dim)

        _, _, past_keys, past_values = ort_session.run(None, {
            "inputs_embeds":  prefix_embeds.numpy(),
            "attention_mask": prefix_mask.numpy(),
            "past_keys":      empty_keys.numpy(),
            "past_values":    empty_values.numpy(),
        })

        # ── run for each new audio chunk ──────────────────────────────────────
        audio_embeds = compute_audio_embeds(model, audio_chunk)   # (B, audio_len, H)
        audio_len    = audio_embeds.shape[1]
        full_mask    = torch.ones(B, prefix_len + audio_len, dtype=torch.long)

        pred_xy, img_logits, _, _ = ort_session.run(None, {
            "inputs_embeds":  audio_embeds.numpy(),
            "attention_mask": full_mask.numpy(),
            "past_keys":      past_keys,    # fixed prefix cache — reused each chunk
            "past_values":    past_values,
        })
        # Discard new_past_keys/values; reuse the same prefix cache next chunk.

    Deployment:
        iOS:     coremltools.convert("score_follower.onnx")
        Android: onnxruntime-android with ORT format conversion
    """
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    model  = ScoreFollowingModel(config)
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_layers  = model.backbone.config.num_hidden_layers
    hidden_size = model.backbone.config.hidden_size
    num_heads   = model.backbone.config.num_key_value_heads
    head_dim    = hidden_size // model.backbone.config.num_attention_heads

    streaming = _StreamingScoreFollower(model)

    # ── Dummy inputs (audio-pass shape; dynamic axes cover prefix pass too) ───
    B       = 1
    seq_len = 32    # audio token count  (dynamic axis: seq_len)
    kv_len  = 64    # prefix KV length   (dynamic axis: kv_len)

    dummy_embeds = torch.randn(B, seq_len, hidden_size)
    dummy_mask   = torch.ones(B, kv_len + seq_len, dtype=torch.long)
    dummy_keys   = torch.zeros(num_layers, B, num_heads, kv_len, head_dim)
    dummy_values = torch.zeros(num_layers, B, num_heads, kv_len, head_dim)

    output_path = os.path.join(output_dir, "score_follower.onnx")
    torch.onnx.export(
        streaming,
        (dummy_embeds, dummy_mask, dummy_keys, dummy_values),
        output_path,
        input_names=["inputs_embeds", "attention_mask", "past_keys", "past_values"],
        output_names=["pred_xy", "img_logits", "new_past_keys", "new_past_values"],
        dynamic_axes={
            "inputs_embeds":   {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "total_len"},
            "past_keys":       {1: "batch", 3: "kv_len"},
            "past_values":     {1: "batch", 3: "kv_len"},
            "pred_xy":         {0: "batch", 1: "seq_len"},
            "img_logits":      {0: "batch", 1: "seq_len"},
            "new_past_keys":   {1: "batch", 3: "new_kv_len"},
            "new_past_values": {1: "batch", 3: "new_kv_len"},
        },
        opset_version=17,
    )
    print(f"Score follower exported → {output_path}")
    print("\nInference flow:")
    print("  1. prefix_embeds = compute_prefix_embeds(model, images, pos, img)  # once")
    print("  2. _, _, keys, vals = ort(prefix_embeds, prefix_mask, empty_kv)   # once")
    print("  3. xy, logits, _, _ = ort(audio_embeds, full_mask, keys, vals)    # per chunk")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score following model training")
    parser.add_argument("--mode", choices=["train", "export"], default="train")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for export")
    parser.add_argument("--output-dir", type=str, default="onnx_export", help="Output dir for ONNX export")
    parser.add_argument("--train-dirs", nargs="+", default=["data/train/*"], help="Training data folders")
    parser.add_argument("--dev-dirs",   nargs="+", default=["data/dev/*"], help="Dev/validation data folders (omit to auto-split 10%% of train)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--audio-length", type=float, default=10.0, help="Audio chunk length in seconds")
    parser.add_argument("--sample-shift", type=float, default=1.0, help="Shift between consecutive training samples (seconds)")

    args = parser.parse_args()

    config = Config()
    config.train_dirs = args.train_dirs
    config.dev_dirs   = args.dev_dirs
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.audio_length_sec = args.audio_length
    config.sample_shift_sec = args.sample_shift

    if args.mode == "train":
        train(config)
    elif args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for export")
        export_onnx(args.checkpoint, args.output_dir)
