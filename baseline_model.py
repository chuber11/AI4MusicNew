"""Lightweight baseline model for score following.

Uses frozen EnCodec-24kHz encoder (audio) + frozen CLIP ViT-B/32 (images)
with a small trainable fusion transformer.

Audio encoder: facebook/encodec-24khz
  - Causal SEANet CNN — no bidirectional attention, no fixed-length limit.
  - Because it is causal, past encoder outputs are immutable: when new audio
    arrives at inference time only new frames need to be encoded (no recompute).
  - Streaming is achieved by keeping a rolling buffer of the last
    `receptive_field_samples` of raw audio; see encode_audio_streaming().
  - Input: 24 kHz mono.  Audio stored at 16 kHz in the dataset is resampled
    to 24 kHz inside the collate function.

Architecture:
  image tokens  (CLIP patches, projected to H=512)
  + pos token   (Fourier-encoded start position, projected to H)
  + page token  (learned embedding, H)
  + bar token   (learned embedding, H)
  + audio tokens (EnCodec encoder frames, projected to H)
  → 6-layer self-attention transformer (H=512, 8 heads)
  → three prediction heads (pos / page / bar)

The model's forward() signature matches ScoreFollowingModel so the same
training loop drives both.  The collate function is accessed via
model.get_collate_fn().
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from model import fourier_encode


# ──────────────────────────────────────────────────────────────────────────────
# Baseline model
# ──────────────────────────────────────────────────────────────────────────────

class BaselineScoreFollowingModel(nn.Module):
    """EnCodec-24kHz encoder + CLIP ViT-B/32 + small fusion transformer.

    Both pretrained encoders are frozen; all projection layers, the fusion
    transformer, and the three heads are trainable (~30 M params).
    """

    HIDDEN   = 512
    N_LAYERS = 6
    N_HEADS  = 8

    # EnCodec-24kHz outputs at 75 Hz; receptive field of the SEANet encoder
    # is ~150 ms ≈ 3600 samples at 24 kHz — used for streaming buffer.
    ENCODER_SR          = 24000
    ENCODER_HZ          = 75          # output frames per second
    STREAMING_BUFFER_MS = 200         # past audio kept for incremental encoding

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import EncodecModel, CLIPVisionModel, CLIPProcessor

        # ── Frozen causal audio encoder (EnCodec-24kHz SEANet CNN) ────────────
        _encodec = EncodecModel.from_pretrained("facebook/encodec-24khz")
        self.audio_encoder = _encodec.encoder   # EncodecEncoder (causal CNN)
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # Determine encoder output channels by a quick dry run
        with torch.no_grad():
            _test = torch.zeros(1, 1, self.ENCODER_SR)   # 1 s of silence
            _out  = self.audio_encoder(_test)
            audio_dim = _out.shape[1]                     # (B, C, T) → C
        print(f"EnCodec encoder output dim: {audio_dim}")

        # ── Frozen vision encoder (CLIP ViT-B/32) ────────────────────────────
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size   # 768
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
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            enc_layer, num_layers=self.N_LAYERS, enable_nested_tensor=False,
        )

        # ── Prediction heads ──────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, inputs, start_pos, start_img, start_bar=None):
        """
        inputs:    dict with keys:
                     audio_24k    (B, 1, T_samples)  mono 24 kHz raw audio
                     pixel_values (B * num_pages, 3, 224, 224)
                     num_pages    int
        start_pos: (B, 2) float, normalized (x, y)
        start_img: (B,)   long, current page index
        start_bar: (B,)   long, current bar value (0=playing, N=rest bar N)

        Returns:
            pos_logits:  (B, T_audio, grid_w * grid_h)
            page_logits: (B, T_audio, max_num_images)
            bar_logits:  (B, T_audio, max_bar)
        """
        inputs.pop("piece_ids", None)

        B      = start_pos.shape[0]
        device = start_pos.device
        H      = self.HIDDEN
        if start_bar is None:
            start_bar = torch.zeros(B, dtype=torch.long, device=device)

        # ── Encode audio (causal CNN — no gradient through frozen encoder) ────
        audio_24k = inputs["audio_24k"].to(device)           # (B, 1, T)
        with torch.no_grad():
            enc_out = self.audio_encoder(audio_24k)           # (B, C, T_enc)
        audio_emb = self.audio_proj(enc_out.transpose(1, 2).float())  # (B, T_enc, H)

        # ── Encode images ─────────────────────────────────────────────────────
        num_pages = inputs["num_pages"]
        pixels    = inputs["pixel_values"].to(device)         # (B*P, 3, 224, 224)
        with torch.no_grad():
            img_patches = self.vision_encoder(pixels).last_hidden_state[:, 1:]
        img_emb   = self.vision_proj(img_patches.float())     # (B*P, n_p, H)
        n_patches = img_emb.shape[1]
        img_emb   = img_emb.reshape(B, num_pages * n_patches, H)

        # ── Context tokens ────────────────────────────────────────────────────
        pos_enc  = fourier_encode(start_pos.float(), self.config.pos_num_freqs)
        pos_emb  = self.pos_proj(pos_enc).unsqueeze(1)
        page_emb = self.page_embed(start_img).unsqueeze(1)
        bar_emb  = self.bar_embed(start_bar).unsqueeze(1)

        # ── Fusion: [img | pos | page | bar | audio] ──────────────────────────
        seq     = torch.cat([img_emb, pos_emb, page_emb, bar_emb, audio_emb], dim=1)
        seq_out = self.fusion(seq)

        audio_start = num_pages * n_patches + 3
        audio_out   = seq_out[:, audio_start:, :]

        pos_logits  = self.pos_head(audio_out)
        page_logits = self.page_head(audio_out)
        bar_logits  = self.bar_head(audio_out)
        return pos_logits, page_logits, bar_logits

    # ──────────────────────────────────────────────────────────────────────────
    # Streaming audio encoding helper
    # ──────────────────────────────────────────────────────────────────────────

    def encode_audio_streaming(self, new_audio_24k: torch.Tensor,
                                buffer: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a new chunk of audio without recomputing past frames.

        Because the EnCodec encoder is a causal CNN, the output for any frame
        depends only on that frame and a fixed window of past samples (the
        receptive field).  Keeping that window as `buffer` lets us encode new
        chunks incrementally.

        Args:
            new_audio_24k: (1, 1, T_new) mono 24 kHz tensor
            buffer:        (1, 1, T_buf) past audio tail, or None on first call

        Returns:
            enc_frames: (1, T_enc_new, C) — new encoder output frames only
            new_buffer: (1, 1, T_buf)     — updated buffer for the next call
        """
        buf_samples = int(self.STREAMING_BUFFER_MS / 1000 * self.ENCODER_SR)
        if buffer is None:
            audio_in = new_audio_24k
        else:
            audio_in = torch.cat([buffer, new_audio_24k], dim=2)

        with torch.no_grad():
            full_enc = self.audio_encoder(audio_in)   # (1, C, T_enc_full)

        # Frames that correspond to new_audio only (last T_enc_new frames)
        T_enc_new = max(1, round(new_audio_24k.shape[2] / self.ENCODER_SR * self.ENCODER_HZ))
        enc_frames = full_enc[:, :, -T_enc_new:].transpose(1, 2)  # (1, T_enc_new, C)

        # Update buffer: keep last buf_samples of raw audio
        tail = audio_in[:, :, -buf_samples:]
        return enc_frames, tail

    # ──────────────────────────────────────────────────────────────────────────
    # Training helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_collate_fn(self, audio_sample_rate):
        return partial(_baseline_collate_fn,
                       model=self,
                       src_sr=audio_sample_rate,
                       dst_sr=self.ENCODER_SR)

    def get_param_groups(self, lr):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]


# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────

def _resample(audio_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio_np
    import librosa
    return librosa.resample(audio_np, orig_sr=src_sr, target_sr=dst_sr)


def _baseline_collate_fn(batch, model, src_sr: int, dst_sr: int):
    """Collate for BaselineScoreFollowingModel.

    Resamples audio from src_sr (dataset rate, e.g. 16 kHz) to dst_sr
    (EnCodec rate, 24 kHz), converts to (B, 1, T) float32 tensors, and
    processes images with the CLIP processor.
    """
    # ── Audio: resample to EnCodec rate ───────────────────────────────────────
    audio_list = [_resample(b["audio"].numpy(), src_sr, dst_sr) for b in batch]
    max_len    = max(a.shape[0] for a in audio_list)
    audio_padded = np.zeros((len(audio_list), max_len), dtype=np.float32)
    for i, a in enumerate(audio_list):
        audio_padded[i, :len(a)] = a
    # (B, 1, T) — EnCodec encoder expects channel dim
    audio_24k = torch.from_numpy(audio_padded).unsqueeze(1)

    # ── Images: CLIP pixel values ─────────────────────────────────────────────
    num_pages = len(batch[0]["all_images"])
    flat_imgs = [img for b in batch for img in b["all_images"]]
    image_inputs = model.clip_processor(images=flat_imgs, return_tensors="pt")

    return {
        # model inputs
        "audio_24k":    audio_24k,
        "pixel_values": image_inputs["pixel_values"],
        "num_pages":    num_pages,
        "piece_ids":    [b["piece_id"] for b in batch],
        # training targets / context
        "start_pos":         torch.stack([b["start_pos"]         for b in batch]),
        "start_img":         torch.stack([b["start_img"]         for b in batch]),
        "start_bar":         torch.stack([b["start_bar"]         for b in batch]),
        "target_pos_patch":  torch.stack([b["target_pos_patch"]  for b in batch]),
        "target_page":       torch.stack([b["target_page"]       for b in batch]),
        "target_bar":        torch.stack([b["target_bar"]        for b in batch]),
    }
