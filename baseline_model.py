"""Lightweight baseline model for score following.

Uses frozen Whisper-small encoder (audio) + frozen CLIP ViT-B/32 (images)
with a small trainable fusion transformer.

~250 M total params, ~30 M trainable — suitable for fast experimentation.

Architecture:
  image tokens  (CLIP patches, projected to H=512)
  + pos token   (Fourier-encoded start position, projected to H)
  + page token  (learned embedding, H)
  + audio tokens (Whisper encoder frames, projected to H)
  → 6-layer self-attention transformer (H=512, 8 heads)
  → three prediction heads (pos / page / bar)

The model's forward() signature matches ScoreFollowingModel so the same
training loop can drive both.  The collate function is different (no Phi-4
processor) and is accessed via model.get_collate_fn().
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import fourier_encode


# ──────────────────────────────────────────────────────────────────────────────
# Baseline model
# ──────────────────────────────────────────────────────────────────────────────

class BaselineScoreFollowingModel(nn.Module):
    """Whisper-small encoder + CLIP ViT-B/32 + small fusion transformer.

    Both encoders are kept frozen.  All projection layers, the fusion
    transformer, and the three heads are trainable.
    """

    HIDDEN    = 512
    N_LAYERS  = 6
    N_HEADS   = 8

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import (
            WhisperModel, WhisperFeatureExtractor,
            CLIPVisionModel, CLIPProcessor,
        )

        # ── Frozen audio encoder (Whisper-small) ─────────────────────────────
        _whisper = WhisperModel.from_pretrained("openai/whisper-small")
        self.audio_encoder = _whisper.encoder   # d_model = 768
        self.audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small"
        )
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # ── Frozen vision encoder (CLIP ViT-B/32) ────────────────────────────
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        audio_dim  = self.audio_encoder.config.d_model           # 768
        vision_dim = self.vision_encoder.config.hidden_size       # 768
        H = self.HIDDEN

        # ── Trainable projections ─────────────────────────────────────────────
        self.audio_proj  = nn.Linear(audio_dim, H)
        self.vision_proj = nn.Linear(vision_dim, H)

        pos_enc_dim = 2 * 2 * config.pos_num_freqs   # 32
        self.pos_proj   = nn.Linear(pos_enc_dim, H, bias=False)
        self.page_embed = nn.Embedding(config.max_num_images, H)
        self.bar_embed  = nn.Embedding(config.max_bar, H)

        # ── Fusion transformer ────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=self.N_HEADS,
            dim_feedforward=H * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable)
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=self.N_LAYERS,
                                            enable_nested_tensor=False)

        # ── Prediction heads (same structure as ScoreFollowingModel) ──────────
        self.pos_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, config.grid_w * config.grid_h),
        )
        self.page_head = nn.Sequential(
            nn.Linear(H, 128),
            nn.GELU(),
            nn.Linear(128, config.max_num_images),
        )
        self.bar_head = nn.Sequential(
            nn.Linear(H, 128),
            nn.GELU(),
            nn.Linear(128, config.max_bar),
        )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Baseline trainable parameters: {n_train:,}")

    def forward(self, inputs, start_pos, start_img, start_bar=None):
        """
        inputs:    dict with keys:
                     audio_features  (B, 80, T_mel)  — log-mel spectrogram
                     pixel_values    (B * num_pages, 3, 224, 224)
                     num_pages       int
        start_pos: (B, 2) float, normalized (x, y)
        start_img: (B,)   long, current page index
        start_bar: (B,)   long, current bar value (0=playing, N=rest bar N)

        Returns:
            pos_logits:  (B, T_audio, grid_w * grid_h)
            page_logits: (B, T_audio, max_num_images)
            bar_logits:  (B, T_audio, max_bar)
        """
        inputs.pop("piece_ids", None)   # not used by baseline

        B      = start_pos.shape[0]
        device = start_pos.device
        H      = self.HIDDEN
        if start_bar is None:
            start_bar = torch.zeros(B, dtype=torch.long, device=device)

        # ── Encode audio ──────────────────────────────────────────────────────
        audio_feat = inputs["audio_features"].to(device)   # (B, 80, T_mel)
        with torch.no_grad():
            audio_hidden = self.audio_encoder(audio_feat).last_hidden_state  # (B, T_a, 768)
        audio_emb = self.audio_proj(audio_hidden.float())                    # (B, T_a, H)

        # ── Encode images ─────────────────────────────────────────────────────
        num_pages = inputs["num_pages"]
        pixels    = inputs["pixel_values"].to(device)  # (B*num_pages, 3, 224, 224)
        with torch.no_grad():
            vision_out = self.vision_encoder(pixels)
            # Drop CLS token, keep patch tokens: (B*num_pages, n_patches, 768)
            img_patches = vision_out.last_hidden_state[:, 1:]
        img_emb = self.vision_proj(img_patches.float())   # (B*num_pages, n_patches, H)
        n_patches = img_emb.shape[1]
        img_emb = img_emb.reshape(B, num_pages * n_patches, H)  # (B, num_pages*n_p, H)

        # ── Position, page, and bar tokens ───────────────────────────────────
        pos_enc  = fourier_encode(start_pos.float(), self.config.pos_num_freqs)
        pos_emb  = self.pos_proj(pos_enc).unsqueeze(1)      # (B, 1, H)
        page_emb = self.page_embed(start_img).unsqueeze(1)  # (B, 1, H)
        bar_emb  = self.bar_embed(start_bar).unsqueeze(1)   # (B, 1, H)

        # ── Concatenate: [img_patches | pos | page | bar | audio] ─────────────
        seq = torch.cat([img_emb, pos_emb, page_emb, bar_emb, audio_emb], dim=1)
        # (B, num_pages*n_patches + 3 + T_a, H)

        # ── Fusion transformer ────────────────────────────────────────────────
        seq_out = self.fusion(seq)   # (B, total_len, H)

        # ── Extract audio portion and apply heads ─────────────────────────────
        audio_start = num_pages * n_patches + 3
        audio_out   = seq_out[:, audio_start:, :]   # (B, T_a, H)

        pos_logits  = self.pos_head(audio_out)    # (B, T_a, grid_w*grid_h)
        page_logits = self.page_head(audio_out)   # (B, T_a, max_num_images)
        bar_logits  = self.bar_head(audio_out)    # (B, T_a, max_bar)
        return pos_logits, page_logits, bar_logits

    def get_collate_fn(self, audio_sample_rate):
        return partial(_baseline_collate_fn,
                       model=self,
                       audio_sample_rate=audio_sample_rate)

    def get_param_groups(self, lr):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return [{"params": trainable, "lr": lr}]


# ──────────────────────────────────────────────────────────────────────────────
# Collate function for the baseline model
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_collate_fn(batch, model, audio_sample_rate):
    """Collate raw audio + PIL images for BaselineScoreFollowingModel.

    Returns a dict whose keys after popping start_pos / start_img / targets are
    {audio_features, pixel_values, num_pages, piece_ids} — passed as `inputs`
    to model.forward().
    """
    # ── Audio → Whisper log-mel spectrogram ───────────────────────────────────
    audios = [b["audio"].numpy() for b in batch]
    audio_inputs = model.audio_feature_extractor(
        audios,
        sampling_rate=audio_sample_rate,
        return_tensors="pt",
        padding="longest",
    )   # input_features: (B, 80, T_mel)

    # ── Images → CLIP pixel values ────────────────────────────────────────────
    # All samples from PieceBatchSampler have the same number of pages.
    num_pages = len(batch[0]["all_images"])
    flat_imgs = [img for b in batch for img in b["all_images"]]
    image_inputs = model.clip_processor(images=flat_imgs, return_tensors="pt")
    # pixel_values: (B * num_pages, 3, 224, 224)

    return {
        # model inputs (remaining after the training loop pops targets)
        "audio_features": audio_inputs["input_features"],
        "pixel_values":   image_inputs["pixel_values"],
        "num_pages":      num_pages,
        "piece_ids":      [b["piece_id"] for b in batch],
        # training targets / context (popped by training loop before model call)
        "start_pos":        torch.stack([b["start_pos"]        for b in batch]),
        "start_img":        torch.stack([b["start_img"]        for b in batch]),
        "start_bar":        torch.stack([b["start_bar"]        for b in batch]),
        "target_pos_patch": torch.stack([b["target_pos_patch"] for b in batch]),
        "target_page":      torch.stack([b["target_page"]      for b in batch]),
        "target_bar":       torch.stack([b["target_bar"]       for b in batch]),
    }
