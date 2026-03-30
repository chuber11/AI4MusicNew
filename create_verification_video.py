#!/usr/bin/env python3
"""
Create a verification video showing interpolated annotations on sheet music images,
synchronized with the audio.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d


def load_annotations(json_path):
    with open(json_path) as f:
        return json.load(f)


def detect_lines(annotations):
    """Detect music lines by finding where x_ratio drops or y_ratio jumps.

    A new line starts when:
      - x_ratio drops significantly (cursor wraps to next line), OR
      - y_ratio jumps significantly (repeat / da capo), OR
      - image_index changes (page turn)

    Bar-annotation entries (image_index == -1) are skipped.
    """
    anns = [a for a in annotations["annotations"] if a["image_index"] != -1]
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


def build_bar_timeline(all_annotations):
    """Build sorted (timestamp_ms, bar_value) events from raw annotation list."""
    normal_anns = sorted(
        [a for a in all_annotations if a["image_index"] != -1],
        key=lambda a: a["timestamp_ms"],
    )
    bar_anns = sorted(
        [a for a in all_annotations
         if a["image_index"] == -1 and a.get("label", "").startswith("bar_")],
        key=lambda a: a["timestamp_ms"],
    )
    events = [(a["timestamp_ms"], 0) for a in normal_anns]
    for a in bar_anns:
        n = int(a["label"].split("_")[1])
        events.append((a["timestamp_ms"], n))
    bar2_times = {a["timestamp_ms"] for a in bar_anns if a["label"] == "bar_2"}
    normal_times = [a["timestamp_ms"] for a in normal_anns]
    for t2 in sorted(bar2_times):
        preceding = [t for t in normal_times if t < t2]
        if preceding:
            events.append((max(preceding) + 1, 1))
    events.sort()
    return events


def get_bar_at_time(t_ms, bar_events):
    """Return bar value at t_ms (0=playing, N=bar N of rest)."""
    bar = 0
    for ts, val in bar_events:
        if ts <= t_ms:
            bar = val
        else:
            break
    return bar


def build_interpolation(lines):
    """Build piecewise linear interpolation with averaged y per line.

    Each line gets its y-coordinates averaged. Between lines the position
    continues rightward to the end of the line, then snaps instantly to
    the start of the next line.
    """
    line_info = []

    for line in lines:
        times = [p["timestamp_ms"] for p in line]
        xs = [p["x_ratio"] for p in line]
        ys = [p["y_ratio"] for p in line]
        img_idx = line[0]["image_index"]
        avg_y = sum(ys) / len(ys)

        line_info.append({
            "start_ms": times[0],
            "end_ms": times[-1],
            "image_index": img_idx,
            "avg_y": avg_y,
            "times": times,
            "xs": xs,
        })

    return line_info


def get_position_at_time(t_ms, line_info):
    """Get interpolated (image_index, x_ratio, y_ratio) at a given time.

    Each line owns the time from its first annotation until just before the
    next line's first annotation. Within annotated points we interpolate x
    linearly; after the last annotation point we extrapolate at the last
    segment's speed (clamped to [0,1]). The cursor then snaps instantly to
    the next line.
    """
    n = len(line_info)

    for i, li in enumerate(line_info):
        # This line owns [its start_ms, next line's start_ms)
        # (or until end of audio for the last line)
        t_start = li["start_ms"]
        t_end = line_info[i + 1]["start_ms"] if i + 1 < n else float("inf")

        if t_start <= t_ms < t_end:
            if len(li["times"]) == 1:
                # Single-point line: extrapolate with no speed info, just hold
                return li["image_index"], li["xs"][0], li["avg_y"]

            if t_ms <= li["end_ms"]:
                # Within annotated range: interpolate
                f = interp1d(li["times"], li["xs"], kind="linear")
                return li["image_index"], float(f(t_ms)), li["avg_y"]
            else:
                # Past last annotation: extrapolate at last segment's speed
                dt_last = li["times"][-1] - li["times"][-2]
                dx_last = li["xs"][-1] - li["xs"][-2]
                speed = dx_last / dt_last if dt_last > 0 else 0
                x = li["xs"][-1] + speed * (t_ms - li["end_ms"])
                x = max(0.0, min(x, 1.0))
                return li["image_index"], x, li["avg_y"]

    # Before first annotation
    li = line_info[0]
    return li["image_index"], li["xs"][0], li["avg_y"]


def create_video(data_dir, output_path="verification_video.mp4", fps=30, ann_file=None):
    # Find files
    data_dir = Path(data_dir)
    if ann_file is None:
        ann_files = list(data_dir.glob("annotations_*.json"))
        if not ann_files:
            raise FileNotFoundError(f"No annotation files found in {data_dir}")
        ann_file = ann_files[0]

    annotations = load_annotations(ann_file)
    audio_file = data_dir / annotations["audio_filename"]
    image_files = [data_dir / f for f in annotations["image_filenames"]]
    audio_duration_ms = annotations["audio_duration_ms"]

    # Load images
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        images.append(img)

    # Build interpolation and bar timeline
    lines      = detect_lines(annotations)
    line_info  = build_interpolation(lines)
    bar_events = build_bar_timeline(annotations["annotations"])

    # Video dimensions: use the max image dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    # Add a header bar for info
    header_h = 60
    frame_h = header_h + max_h
    frame_w = max_w

    # Write video frames to temp file (no audio)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_w, frame_h))

    total_frames = int(audio_duration_ms / 1000 * fps)
    print(f"Generating {total_frames} frames at {fps} fps...")

    # Precompute annotation lists for drawing
    orig_annotations = annotations["annotations"]
    # Build a map: timestamp_ms -> bar label for bar annotations, used to draw markers
    bar_annotation_times = {
        a["timestamp_ms"]: a["label"]
        for a in orig_annotations
        if a["image_index"] == -1 and a.get("label", "").startswith("bar_")
    }

    for frame_idx in range(total_frames):
        t_ms = frame_idx / fps * 1000

        img_idx, x_ratio, y_ratio = get_position_at_time(t_ms, line_info)
        bar_val = get_bar_at_time(t_ms, bar_events)

        # Create frame
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Draw header — highlight in orange during rests
        header_color = (0, 80, 160) if bar_val == 0 else (0, 100, 200)
        cv2.rectangle(frame, (0, 0), (frame_w, header_h), header_color, -1)

        bar_str = "playing" if bar_val == 0 else f"REST bar {bar_val}"
        time_str = (
            f"Time: {t_ms/1000:.2f}s  |  Page: {img_idx+1}/{len(images)}"
            f"  |  x={x_ratio:.3f}  y={y_ratio:.3f}  |  {bar_str}"
        )
        text_color = (255, 255, 255) if bar_val == 0 else (0, 200, 255)
        cv2.putText(frame, time_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        # Draw image
        img = images[img_idx].copy()
        ih, iw = img.shape[:2]

        # Draw all annotation points for this image (small dots)
        for a in orig_annotations:
            if a["image_index"] == img_idx:
                ax = int(a["x_ratio"] * iw)
                ay = int(a["y_ratio"] * ih)
                cv2.circle(img, (ax, ay), 4, (200, 200, 200), -1)

        # Draw line-averaged y positions and line segments
        for li in line_info:
            if li["image_index"] == img_idx:
                avg_py = int(li["avg_y"] * ih)
                pts = [(int(x * iw), avg_py) for x in li["xs"]]
                for j in range(len(pts) - 1):
                    cv2.line(img, pts[j], pts[j + 1], (0, 180, 0), 2)
                for pt in pts:
                    cv2.circle(img, pt, 5, (0, 255, 0), -1)

        # Draw bar-annotation timestamps as vertical cyan lines at image bottom
        # Show bars within ±5 seconds of current time
        for bar_t_ms, bar_label in bar_annotation_times.items():
            if abs(bar_t_ms - t_ms) < 5000:
                # Map time to x position within the image (rough linear mapping)
                x_frac = (bar_t_ms - (t_ms - 5000)) / 10000.0
                bx = int(x_frac * iw)
                if 0 <= bx < iw:
                    cv2.line(img, (bx, ih - 30), (bx, ih), (255, 255, 0), 2)
                    cv2.putText(img, bar_label, (bx + 2, ih - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw current interpolated position — cyan crosshair during rest, red during playing
        cx = int(x_ratio * iw)
        cy = int(y_ratio * ih)
        cursor_color = (0, 0, 255) if bar_val == 0 else (0, 200, 255)
        cv2.circle(img, (cx, cy), 15, cursor_color, 3)
        cv2.line(img, (cx - 20, cy), (cx + 20, cy), cursor_color, 2)
        cv2.line(img, (cx, cy - 20), (cx, cy + 20), cursor_color, 2)

        # Place image in frame
        frame[header_h:header_h + ih, :iw] = img
        writer.write(frame)

        if frame_idx % (fps * 10) == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({t_ms/1000:.1f}s)")

    writer.release()
    print("Frames written. Muxing with audio...")

    # Combine video with audio using ffmpeg
    output_path = str(Path(output_path).resolve())
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", str(audio_file),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(temp_video_path)
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="data/Hands_Across_the_Sea")
    parser.add_argument("output",   nargs="?", default="verification_video.mp4")
    parser.add_argument("--ann-file", default=None,
                        help="Annotation JSON to use (default: annotations_*.json in data_dir)")
    args = parser.parse_args()
    create_video(args.data_dir, args.output, ann_file=args.ann_file)
