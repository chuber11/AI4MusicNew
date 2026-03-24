
  1. create_verification_video.py — Verification Video

  - Line detection: Groups annotations into lines by detecting x_ratio drops (new staff line)
  - Y-averaging: Averages all y-coordinates within each detected line
  - Interpolation: Linear interpolation of x within each line; smooth transitions between lines
  - Video output: Shows the sheet music image with:
    - Gray dots = original annotation points
    - Green dots + lines = interpolated path with averaged y
    - Red crosshair = current position moving in real-time
    - Header bar with timestamp and coordinates
  - Audio is muxed in via ffmpeg

  Run: python create_verification_video.py

  2. train.py — Training Pipeline (3 stages)

  Stage 1: Fine-tune Phi-4-multimodal-instruct

  - Uses Phi-4's native image + audio multimodal processing
  - LoRA adapters (r=16) on attention layers for efficient fine-tuning
  - Custom regression head predicts next 10 positions (x, y, same_image_flag) at 500ms intervals
  - Input: sheet music image + 5s audio segment + start position
  - Loss: MSE on coordinates + BCE on page-transition flag
  - Samples every 250ms along the interpolated trajectory (~600 training samples from this piece)

  Run: python train.py --mode train

  Stage 2: Knowledge distillation for mobile

  - Student model (~45M params): Whisper-tiny (audio) + MobileNetV3-small (image) + lightweight head
  - Distills from the Phi-4 teacher model
  - Combined distillation + ground-truth loss

  Run: python train.py --mode distill --checkpoint checkpoints/best_model.pt

  Stage 3: Mobile export

  - Exports to ONNX, convertible to CoreML (iOS) or ORT Mobile (Android)

  Run: python train.py --mode export --checkpoint checkpoints/best_model.pt

