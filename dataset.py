"""Dataset and data utilities for score-following training."""

import glob as _glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, Sampler


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
      - Target: dense positions across the audio segment
    """

    def __init__(self, data_dirs, config, processor=None):
        self.config = config
        self.processor = processor
        self.samples = []

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

        if sr != self.config.audio_sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.config.audio_sample_rate)

        min_t = line_info[0]["start_ms"]
        max_t = line_info[-1]["end_ms"]
        audio_len_ms = self.config.audio_length_sec * 1000
        shift_ms = int(self.config.sample_shift_sec * 1000)

        start_times = list(range(int(min_t), int(max_t - audio_len_ms), shift_ms))

        _N_DENSE = 200

        for t_start in start_times:
            if t_start + audio_len_ms > audio_duration_ms:
                continue

            img_idx_start, x_start, y_start = get_position_at_time(t_start, line_info)

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
                "piece_id": str(data_dir),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cfg = self.config

        t_start_sec = sample["t_start_ms"] / 1000.0
        sr = sample["audio_sr"]
        start_sample = int(t_start_sec * sr)
        end_sample = start_sample + int(cfg.audio_length_sec * sr)
        audio_segment = sample["audio_data"][start_sample:end_sample]

        expected_len = int(cfg.audio_length_sec * sr)
        if len(audio_segment) < expected_len:
            audio_segment = np.pad(audio_segment, (0, expected_len - len(audio_segment)))

        from PIL import Image
        all_images = []
        for p in sample["image_paths"]:
            img = Image.open(str(p)).convert("RGB")
            w, h = img.size
            img = img.resize((cfg.image_width, int(h * cfg.image_width / w)), Image.LANCZOS)
            all_images.append(img)

        target_xy = torch.tensor(
            [(t[0], t[1]) for t in sample["targets"]], dtype=torch.float32,
        )
        target_img = torch.tensor(
            [t[2] for t in sample["targets"]], dtype=torch.long,
        )
        start_pos = torch.tensor(
            [sample["start_pos"][0], sample["start_pos"][1]], dtype=torch.float32,
        )
        start_img = torch.tensor(sample["start_pos"][2], dtype=torch.long)

        return {
            "audio": torch.tensor(audio_segment, dtype=torch.float32),
            "all_images": all_images,
            "start_pos": start_pos,
            "start_img": start_img,
            "target_xy": target_xy,
            "target_img": target_img,
            "piece_id": sample["piece_id"],
        }


def collate_fn(batch, processor, audio_sample_rate):
    """Run the processor on the whole batch inside the DataLoader worker."""
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

    inputs["start_pos"]  = torch.stack([b["start_pos"]  for b in batch])
    inputs["start_img"]  = torch.stack([b["start_img"]  for b in batch])
    inputs["target_xy"]  = torch.stack([b["target_xy"]  for b in batch])
    inputs["target_img"] = torch.stack([b["target_img"] for b in batch])
    inputs["piece_ids"]  = [b["piece_id"] for b in batch]
    return inputs


class PieceBatchSampler(Sampler):
    """Yields batches where every sample comes from the same music piece.

    This enables CachingImageEmbed to skip the vision encoder for all but the
    first sample of each piece, since the sheet music images never change.

    Incomplete batches (pieces with fewer than batch_size samples) are dropped
    by default.  Pass drop_last=False to keep them.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = True):
        # Support both Dataset (has .samples) and Subset (has .indices + .dataset)
        if hasattr(dataset, "indices"):
            all_samples = [dataset.dataset.samples[i] for i in dataset.indices]
        else:
            all_samples = dataset.samples

        piece_to_indices: dict = defaultdict(list)
        for local_idx, s in enumerate(all_samples):
            piece_to_indices[s["piece_id"]].append(local_idx)

        self.batches = []
        for indices in piece_to_indices.values():
            indices = list(indices)
            if shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                if drop_last and len(chunk) < batch_size:
                    continue
                self.batches.append(chunk)

        self._shuffle = shuffle
        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)
