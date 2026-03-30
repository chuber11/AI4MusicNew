"""Score-following model, loss, and ONNX export."""

import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Token ID used by Phi-4 for image-patch positions in the sequence
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>' in Phi-4's vocab


# ──────────────────────────────────────────────────────────────────────────────
# Cached image embedding
# ──────────────────────────────────────────────────────────────────────────────

class CachingImageEmbed(nn.Module):
    """Wraps Phi4MMImageEmbedding to cache per-piece image token embeddings.

    The vision encoder (ViT + projection) is entirely frozen, so its output is
    identical for every training sample that shares the same music piece.  This
    wrapper runs the encoder once per piece, stores only the compact image-token
    slice (shape: [n_img_tokens, hidden_size] on CPU), and on subsequent calls
    reconstructs image_hidden_states from the cache without touching the ViT.

    Usage:
        Set  wrapper.current_piece_ids = ["piece_a", "piece_a", ...]
        (length == batch_size) before each backbone forward call.
        Leave as None to disable caching (e.g. during ONNX export).
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._cache: dict = {}          # piece_id -> Tensor [n_img, H] on CPU
        self.current_piece_ids = None   # list[str] | None

    def forward(self, input_ids, input_embeds, image_sizes=None, wte=None, **kwargs):
        if self.current_piece_ids is None:
            return self.original(
                input_ids, input_embeds, image_sizes=image_sizes, wte=wte, **kwargs
            )

        ids = self.current_piece_ids
        img_mask = (input_ids == _IMAGE_SPECIAL_TOKEN_ID)   # (B, seq_len) bool

        # ── Cache miss: run the vision encoder, store result ──────────────────
        if any(pid not in self._cache for pid in ids):
            with torch.no_grad():
                full_out = self.original(
                    input_ids, input_embeds, image_sizes=image_sizes, wte=wte, **kwargs
                )
            for i, pid in enumerate(ids):
                if pid not in self._cache:
                    self._cache[pid] = full_out[i][img_mask[i]].detach().cpu()
            return full_out

        # ── Cache hit: reconstruct without running the ViT ────────────────────
        assert wte is not None, "wte must be provided for cache-hit reconstruction"
        with torch.no_grad():
            hidden = wte(input_ids).detach().clone()    # (B, seq_len, H)
        dev, dtype = hidden.device, hidden.dtype
        for i, pid in enumerate(ids):
            hidden[i][img_mask[i]] = self._cache[pid].to(device=dev, dtype=dtype)
        return hidden

    def clear_cache(self):
        self._cache.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Fourier positional encoding
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


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

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
    - Three heads (shared hidden representations):
        pos_head:  predicts patch within current page  (grid_w × grid_h classes)
        page_head: predicts which page                 (max_num_images classes)
        bar_head:  predicts rest bar (0=playing, N=bar N of rest)  (max_bar classes)
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
            {"additional_special_tokens": ["<|pos|>", "<|page|>", "<|bar|>"]}
        )
        self.pos_token_id  = self.processor.tokenizer.convert_tokens_to_ids("<|pos|>")
        self.page_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|page|>")
        self.bar_token_id  = self.processor.tokenizer.convert_tokens_to_ids("<|bar|>")

        # ── Backbone: load with native vision-lora and speech-lora ───────────
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
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.backbone.resize_token_embeddings(len(self.processor.tokenizer))

        # ── Install caching wrapper around the frozen vision encoder ──────────
        _embed_ext = self.backbone.model.embed_tokens_extend
        self._img_cache = CachingImageEmbed(_embed_ext.image_embed)
        _embed_ext.image_embed = self._img_cache

        hidden_size = self.backbone.config.hidden_size  # 3072
        self._hidden_size = hidden_size

        # ── Freeze everything; unfreeze speech-LoRA only ─────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_adapters = set()
        for name, _ in self.backbone.named_parameters():
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ("lora_A", "lora_B") and i + 1 < len(parts):
                    lora_adapters.add(parts[i + 1])

        speech_adapters = lora_adapters - {"vision"}
        n_speech = 0
        for name, param in self.backbone.named_parameters():
            if any(f".{a}." in name for a in speech_adapters):
                param.requires_grad = True
                n_speech += param.numel()
        print(f"Trainable speech-LoRA parameters: {n_speech:,}")

        # ── Position, page, and bar token projections (always trainable) ──────
        _device = next(self.backbone.parameters()).device
        pos_enc_dim = 2 * 2 * config.pos_num_freqs      # 32 for default 8 freqs
        self.pos_proj  = nn.Linear(pos_enc_dim, hidden_size, bias=False).to(_device)
        self.page_proj = nn.Embedding(config.max_num_images, hidden_size).to(_device)
        self.bar_proj  = nn.Embedding(config.max_bar,        hidden_size).to(_device)

        # ── Three prediction heads — one per output task ─────────────────────
        # pos_head:  which patch within the current page?  (grid_w * grid_h classes)
        # page_head: which page?                           (max_num_images classes)
        # bar_head:  which rest bar? (0=playing)           (max_bar classes)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, config.grid_w * config.grid_h),
        ).to(_device)
        self.page_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, config.max_num_images),
        ).to(_device)
        self.bar_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, config.max_bar),
        ).to(_device)

        # ── Permanent injection hook ───────────────────────────────────────────
        self._pos_emb   = None
        self._page_emb  = None
        self._bar_emb   = None
        self._pos_locs  = None
        self._page_locs = None
        self._bar_locs  = None

        def _permanent_inject(module, args, kwargs):
            if self._pos_emb is None:
                return args, kwargs
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return args, kwargs
            hidden = hidden.clone()
            for row in self._pos_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = self._pos_emb[b]
            for row in self._page_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = self._page_emb[b]
            for row in self._bar_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = self._bar_emb[b]
            if args:
                return (hidden,) + args[1:], kwargs
            return args, {**kwargs, "hidden_states": hidden}

        self.backbone.model.layers[0].register_forward_pre_hook(
            _permanent_inject, with_kwargs=True,
        )

    def forward(self, inputs, start_pos, start_img, start_bar=None):
        """
        inputs:     dict   processor output (input_ids, pixel_values, etc.) on device
        start_pos:  (B, 2) normalized (x, y) ∈ [0, 1]
        start_img:  (B,)   long, current page index
        start_bar:  (B,)   long, current bar value (0=playing, N=rest bar N)

        Returns:
            pos_logits:  (B, seq_len, grid_w * grid_h)
            page_logits: (B, seq_len, max_num_images)
            bar_logits:  (B, seq_len, max_bar)
        """
        self._img_cache.current_piece_ids = inputs.pop("piece_ids", None)
        if start_bar is None:
            start_bar = torch.zeros(start_img.shape[0], dtype=torch.long, device=start_img.device)

        pos_enc  = fourier_encode(start_pos, self.config.pos_num_freqs)
        self._pos_emb   = self.pos_proj(pos_enc.float()).to(torch.bfloat16)
        self._page_emb  = self.page_proj(start_img).to(torch.bfloat16)
        self._bar_emb   = self.bar_proj(start_bar).to(torch.bfloat16)
        self._pos_locs  = (inputs["input_ids"] == self.pos_token_id ).nonzero(as_tuple=False)
        self._page_locs = (inputs["input_ids"] == self.page_token_id).nonzero(as_tuple=False)
        self._bar_locs  = (inputs["input_ids"] == self.bar_token_id ).nonzero(as_tuple=False)

        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        last_hidden  = outputs.hidden_states[-1].float()   # (B, seq_len, H)
        pos_logits   = self.pos_head(last_hidden)           # (B, seq_len, grid_w*grid_h)
        page_logits  = self.page_head(last_hidden)          # (B, seq_len, max_num_images)
        bar_logits   = self.bar_head(last_hidden)           # (B, seq_len, max_bar)
        return pos_logits, page_logits, bar_logits

    def get_collate_fn(self, audio_sample_rate):
        from dataset import collate_fn
        return partial(collate_fn,
                       processor=self.processor,
                       audio_sample_rate=audio_sample_rate)

    def get_param_groups(self, lr):
        speech_lora = [p for p in self.backbone.parameters() if p.requires_grad]
        other = (
            list(self.pos_proj.parameters())
            + list(self.page_proj.parameters())
            + list(self.bar_proj.parameters())
            + list(self.pos_head.parameters())
            + list(self.page_head.parameters())
            + list(self.bar_head.parameters())
        )
        return [{"params": speech_lora, "lr": lr}, {"params": other, "lr": lr}]


# ──────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ──────────────────────────────────────────────────────────────────────────────

def xy_to_pos_patch_index(x, y, grid_w, grid_h):
    """Convert (x, y) to a flat patch index within a page.

    Works with scalars (returns int) or tensors (returns long tensor).
    """
    if isinstance(x, torch.Tensor):
        col = (x * grid_w).long().clamp(0, grid_w - 1)
        row = (y * grid_h).long().clamp(0, grid_h - 1)
        return row * grid_w + col
    col = min(int(x * grid_w), grid_w - 1)
    row = min(int(y * grid_h), grid_h - 1)
    return row * grid_w + col


def logits_to_position(pos_logits, page_logits, bar_logits, grid_w, grid_h):
    """Convert three-head logits to (x, y, page, bar) predictions.

    pos_logits:  (..., grid_w * grid_h)
    page_logits: (..., max_num_images)
    bar_logits:  (..., max_bar)

    Returns: x (...), y (...), page (...), bar (...) as tensors.
    x, y are softmax-weighted patch-center averages (sub-patch precision).
    page and bar are argmax predictions.
    """
    probs = torch.softmax(pos_logits.float(), dim=-1)
    rows  = torch.arange(grid_h, device=pos_logits.device)
    cols  = torch.arange(grid_w, device=pos_logits.device)
    r, c  = torch.meshgrid(rows, cols, indexing="ij")
    center_x = (c.flatten().float() + 0.5) / grid_w
    center_y = (r.flatten().float() + 0.5) / grid_h

    x    = (probs * center_x).sum(dim=-1)
    y    = (probs * center_y).sum(dim=-1)
    page = page_logits.argmax(dim=-1)
    bar  = bar_logits.argmax(dim=-1)
    return x, y, page, bar


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def score_following_loss(pos_logits, page_logits, bar_logits,
                         target_pos_patch, target_page, target_bar):
    """Three-head CE loss with position masking during rest bars.

    pos_logits:      (B, seq_len, grid_w * grid_h)
    page_logits:     (B, seq_len, max_num_images)
    bar_logits:      (B, seq_len, max_bar)
    target_pos_patch:(B, num_dense)  patch index within page
    target_page:     (B, num_dense)  page index
    target_bar:      (B, num_dense)  bar value (0=playing, N=rest bar N)

    Position CE is only computed for frames where target_bar == 0 (playing).
    Page and bar CE are computed for all frames.

    Returns:
        total_loss (scalar), pos_acc, page_acc, bar_acc (scalars)
    """
    B, seq_len, C_pos  = pos_logits.shape
    C_page = page_logits.shape[-1]
    C_bar  = bar_logits.shape[-1]

    def _interp(t):
        if t.shape[1] != seq_len:
            t = F.interpolate(
                t.float().unsqueeze(1), size=seq_len, mode="nearest",
            ).squeeze(1).long()
        return t

    target_pos_patch = _interp(target_pos_patch)
    target_page      = _interp(target_page)
    target_bar       = _interp(target_bar)

    # ── Bar loss (always) ─────────────────────────────────────────────────────
    flat_bar_logits  = bar_logits.reshape(-1, C_bar)
    flat_bar_targets = target_bar.reshape(-1)
    bar_loss = F.cross_entropy(flat_bar_logits, flat_bar_targets)
    bar_acc  = (flat_bar_logits.argmax(-1) == flat_bar_targets).float().mean()

    # ── Page loss (always) ────────────────────────────────────────────────────
    flat_page_logits  = page_logits.reshape(-1, C_page)
    flat_page_targets = target_page.reshape(-1)
    page_loss = F.cross_entropy(flat_page_logits, flat_page_targets)
    page_acc  = (flat_page_logits.argmax(-1) == flat_page_targets).float().mean()

    # ── Position loss (all frames) ────────────────────────────────────────────
    flat_pos_logits  = pos_logits.reshape(-1, C_pos)
    flat_pos_targets = target_pos_patch.reshape(-1)
    pos_loss = F.cross_entropy(flat_pos_logits, flat_pos_targets)
    pos_acc  = (flat_pos_logits.argmax(-1) == flat_pos_targets).float().mean()

    total_loss = pos_loss + page_loss + bar_loss
    return total_loss, pos_acc, page_acc, bar_acc


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export — single file, LLM weights stored once
# ──────────────────────────────────────────────────────────────────────────────

class _StreamingScoreFollower(nn.Module):
    """Single ONNX graph handling both prefix and audio-decode passes."""

    def __init__(self, model: ScoreFollowingModel):
        super().__init__()
        self.transformer = model.backbone.model
        self.pos_head    = model.pos_head
        self.page_head   = model.page_head
        self.bar_head    = model.bar_head

    def forward(
        self,
        inputs_embeds:  torch.Tensor,
        attention_mask: torch.Tensor,
        past_keys:      torch.Tensor,
        past_values:    torch.Tensor,
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

        new_past_keys   = torch.stack([kv[0] for kv in out.past_key_values])
        new_past_values = torch.stack([kv[1] for kv in out.past_key_values])

        h = out.last_hidden_state.float()
        pos_logits  = self.pos_head(h)
        page_logits = self.page_head(h)
        bar_logits  = self.bar_head(h)

        return pos_logits, page_logits, bar_logits, new_past_keys, new_past_values


def _capture_inputs_embeds(model: ScoreFollowingModel, backbone_inputs: dict) -> torch.Tensor:
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

    return captured["embeds"]


def compute_prefix_embeds(
    model:     ScoreFollowingModel,
    images:    list,
    start_pos: torch.Tensor,
    start_img: torch.Tensor,
) -> torch.Tensor:
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

    pos_enc  = fourier_encode(start_pos, cfg.pos_num_freqs)
    pos_emb  = model.pos_proj(pos_enc.float()).to(torch.bfloat16)
    page_emb = model.page_proj(start_img).to(torch.bfloat16)

    input_ids = inputs["input_ids"]
    pos_locs  = (input_ids == model.pos_token_id ).nonzero(as_tuple=False)
    page_locs = (input_ids == model.page_token_id).nonzero(as_tuple=False)
    injected  = [False]
    captured  = {}

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

    return captured["embeds"]


def compute_audio_embeds(
    model: ScoreFollowingModel,
    audio: torch.Tensor,
) -> torch.Tensor:
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

    return _capture_inputs_embeds(model, audio_inputs)


def export_onnx(config, checkpoint_path, output_dir="onnx_export"):
    """Export as a single ONNX graph."""
    os.makedirs(output_dir, exist_ok=True)

    model  = ScoreFollowingModel(config)
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    key    = "trainable_state_dict" if "trainable_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key], strict=False)
    model.eval()

    num_layers  = model.backbone.config.num_hidden_layers
    hidden_size = model.backbone.config.hidden_size
    num_heads   = model.backbone.config.num_key_value_heads
    head_dim    = hidden_size // model.backbone.config.num_attention_heads

    streaming = _StreamingScoreFollower(model)

    B       = 1
    seq_len = 32
    kv_len  = 64

    dummy_embeds = torch.randn(B, seq_len, hidden_size)
    dummy_mask   = torch.ones(B, kv_len + seq_len, dtype=torch.long)
    dummy_keys   = torch.zeros(num_layers, B, num_heads, kv_len, head_dim)
    dummy_values = torch.zeros(num_layers, B, num_heads, kv_len, head_dim)

    output_path = os.path.join(output_dir, "score_follower.onnx")
    import torch.onnx
    torch.onnx.export(
        streaming,
        (dummy_embeds, dummy_mask, dummy_keys, dummy_values),
        output_path,
        input_names=["inputs_embeds", "attention_mask", "past_keys", "past_values"],
        output_names=["pos_logits", "page_logits", "bar_logits", "new_past_keys", "new_past_values"],
        dynamic_axes={
            "inputs_embeds":   {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "total_len"},
            "past_keys":       {1: "batch", 3: "kv_len"},
            "past_values":     {1: "batch", 3: "kv_len"},
            "pos_logits":      {0: "batch", 1: "seq_len"},
            "page_logits":     {0: "batch", 1: "seq_len"},
            "bar_logits":      {0: "batch", 1: "seq_len"},
            "new_past_keys":   {1: "batch", 3: "new_kv_len"},
            "new_past_values": {1: "batch", 3: "new_kv_len"},
        },
        opset_version=17,
    )
    print(f"Score follower exported → {output_path}")
