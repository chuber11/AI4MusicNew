"""Score-following model, loss, and ONNX export."""

import math
import os

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

        # ── Install caching wrapper around the frozen vision encoder ──────────
        _embed_ext = self.backbone.model.embed_tokens_extend
        self._img_cache = CachingImageEmbed(_embed_ext.image_embed)
        _embed_ext.image_embed = self._img_cache

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
        #print(f"All LoRA adapters found : {sorted(lora_adapters)}")
        #print(f"Adapters to fine-tune   : {sorted(speech_adapters)}")

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

        # ── MLP head — one patch prediction per token ────────────────────────
        num_patches = config.grid_w * config.grid_h * config.max_num_images
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, num_patches),
        ).to(_device)

        # ── Permanent injection hook ───────────────────────────────────────────
        # Registered once in __init__ so it fires during gradient checkpointing
        # recomputation (which re-runs layers[0] during backward).  The actual
        # embeddings and token locations are stored as instance attributes and
        # updated at the start of each forward() call.
        self._pos_emb   = None
        self._page_emb  = None
        self._pos_locs  = None
        self._page_locs = None

        def _permanent_inject(module, args, kwargs):
            if self._pos_emb is None:
                return args, kwargs
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return args, kwargs
            # Clone before writing to avoid in-place modification errors with
            # gradient checkpointing (saved tensor version must not change).
            hidden = hidden.clone()
            for row in self._pos_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = self._pos_emb[b]
            for row in self._page_locs:
                b, s = row[0].item(), row[1].item()
                hidden[b, s] = self._page_emb[b]
            if args:
                return (hidden,) + args[1:], kwargs
            return args, {**kwargs, "hidden_states": hidden}

        self.backbone.model.layers[0].register_forward_pre_hook(
            _permanent_inject, with_kwargs=True,
        )

    def forward(self, inputs, start_pos, start_img):
        """
        inputs:     dict   processor output (input_ids, pixel_values, etc.) on device
        start_pos:  (B, 2) normalized (x, y) ∈ [0, 1]
        start_img:  (B,)   long, current page index

        Returns:
            patch_logits: (B, seq_len, num_patches) raw logits over grid patches
        """
        # ── Set piece IDs for the image cache (popped before backbone call) ─────
        self._img_cache.current_piece_ids = inputs.pop("piece_ids", None)

        # ── Compute pos/page token embeddings and store for hook ──────────────
        pos_enc  = fourier_encode(start_pos, self.config.pos_num_freqs)     # (B, 32)
        self._pos_emb   = self.pos_proj(pos_enc.float()).to(torch.bfloat16) # (B, H)
        self._page_emb  = self.page_proj(start_img).to(torch.bfloat16)     # (B, H)
        self._pos_locs  = (inputs["input_ids"] == self.pos_token_id ).nonzero(as_tuple=False)
        self._page_locs = (inputs["input_ids"] == self.page_token_id).nonzero(as_tuple=False)

        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # ── Apply head at every sequence position ─────────────────────────────
        last_hidden  = outputs.hidden_states[-1].float()   # (B, seq_len, H)
        patch_logits = self.head(last_hidden)              # (B, seq_len, num_patches)
        return patch_logits


# ──────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ──────────────────────────────────────────────────────────────────────────────

def xy_to_patch_index(x, y, page, grid_w, grid_h):
    """Convert (x, y, page) to a flat patch index.

    Works with scalars (returns int) or tensors (returns long tensor).
    """
    if isinstance(x, torch.Tensor):
        col = (x * grid_w).long().clamp(0, grid_w - 1)
        row = (y * grid_h).long().clamp(0, grid_h - 1)
        return page.long() * (grid_w * grid_h) + row * grid_w + col
    col = min(int(x * grid_w), grid_w - 1)
    row = min(int(y * grid_h), grid_h - 1)
    return int(page) * grid_w * grid_h + row * grid_w + col


def logits_to_position(logits, grid_w, grid_h, num_pages):
    """Convert patch logits to (x, y, page) via softmax-weighted average.

    logits: (..., num_patches)
    Returns: x (...), y (...), page (...) as float tensors
    """
    probs = torch.softmax(logits.float(), dim=-1)
    device = logits.device

    # Build patch center coordinates: (num_patches,)
    pages = torch.arange(num_pages, device=device)
    rows  = torch.arange(grid_h, device=device)
    cols  = torch.arange(grid_w, device=device)
    p, r, c = torch.meshgrid(pages, rows, cols, indexing="ij")
    center_x = (c.flatten().float() + 0.5) / grid_w
    center_y = (r.flatten().float() + 0.5) / grid_h
    center_p = p.flatten().float()

    x    = (probs * center_x).sum(dim=-1)
    y    = (probs * center_y).sum(dim=-1)
    # Page: marginalize over grid cells per page, then argmax
    per_page = probs.unflatten(-1, (num_pages, grid_h * grid_w)).sum(dim=-1)
    page = per_page.argmax(dim=-1)

    return x, y, page


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def score_following_loss(patch_logits, target_patches):
    """Per-token CE loss: interpolate dense targets to match actual sequence length.

    patch_logits:   (B, seq_len, num_patches) predicted logits at every token
    target_patches: (B, num_dense)            ground truth patch indices (long)

    Returns:
        loss (scalar)
    """
    B, seq_len, C = patch_logits.shape
    num_dense = target_patches.shape[1]

    if num_dense != seq_len:
        target_patches = F.interpolate(
            target_patches.float().unsqueeze(1),  # (B, 1, num_dense)
            size=seq_len,
            mode="nearest",
        ).squeeze(1).long()                       # (B, seq_len)

    return F.cross_entropy(
        patch_logits.reshape(-1, C),
        target_patches.reshape(-1),
    )


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

        patch_logits = self.head(out.last_hidden_state.float())  # (B, seq_len, num_patches)

        return patch_logits, new_past_keys, new_past_values


def _capture_inputs_embeds(model: ScoreFollowingModel, backbone_inputs: dict) -> torch.Tensor:
    """Run the backbone up to (but not including) layer 0, capturing inputs_embeds."""
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


def export_onnx(config, checkpoint_path, output_dir="onnx_export"):
    """Export as a single ONNX graph — LLM weights stored exactly once.

    File: score_follower.onnx
        Inputs:
            inputs_embeds   (B, seq_len, H)                 pre-computed embeddings
            attention_mask  (B, kv_len + seq_len)
            past_keys       (L, B, heads, kv_len, head_dim)
            past_values     (L, B, heads, kv_len, head_dim)
        Outputs:
            patch_logits    (B, seq_len, num_patches)
            new_past_keys   (L, B, heads, kv_len + seq_len, head_dim)
            new_past_values (L, B, heads, kv_len + seq_len, head_dim)
    """
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
        output_names=["patch_logits", "new_past_keys", "new_past_values"],
        dynamic_axes={
            "inputs_embeds":   {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "total_len"},
            "past_keys":       {1: "batch", 3: "kv_len"},
            "past_values":     {1: "batch", 3: "kv_len"},
            "patch_logits":    {0: "batch", 1: "seq_len"},
            "new_past_keys":   {1: "batch", 3: "new_kv_len"},
            "new_past_values": {1: "batch", 3: "new_kv_len"},
        },
        opset_version=17,
    )
    print(f"Score follower exported → {output_path}")
    print("\nInference flow:")
    print("  1. prefix_embeds = compute_prefix_embeds(model, images, pos, img)  # once")
    print("  2. _, keys, vals = ort(prefix_embeds, prefix_mask, empty_kv)      # once")
    print("  3. logits, _, _  = ort(audio_embeds, full_mask, keys, vals)       # per chunk")
    print("  4. x, y, page   = logits_to_position(logits, grid_w, grid_h, N)  # decode")
