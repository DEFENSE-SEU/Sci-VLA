#!/usr/bin/env python
"""Annotate every video frame with task-completion and switch predictions."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

from model import load_checkpoint


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_tensor(processor, rgb_frame, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(rgb_frame)
    return processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", required=True, help="Task description or explicit success condition.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Override the validation-selected threshold.")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--required-positive", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if not 1 <= args.required_positive <= args.window_size:
        raise ValueError("--required-positive must be in [1, window-size]")

    device = _resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    processor = AutoProcessor.from_pretrained(checkpoint["model_name"])
    threshold = (
        float(checkpoint["threshold"])
        if args.threshold is None
        else float(args.threshold)
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output video: {args.output}")

    ok, first_bgr = capture.read()
    if not ok:
        capture.release()
        writer.release()
        raise RuntimeError(f"Input video has no frames: {args.video}")

    text_inputs = processor.tokenizer(
        [args.text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    initial_pixels = _image_tensor(processor, first_rgb, device)
    with torch.inference_mode():
        initial_feature = model.encode_image(initial_pixels)
        text_feature = model.encode_text(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )

    history: deque[bool] = deque(maxlen=args.window_size)
    frame_index = 0
    current_bgr = first_bgr
    while True:
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
        current_pixels = _image_tensor(processor, current_rgb, device)
        with torch.inference_mode():
            current_feature = model.encode_image(current_pixels)
            logit = model.classify_features(
                current_feature,
                initial_feature,
                text_feature,
            )
            probability = float(torch.sigmoid(logit).item())

        raw_complete = probability >= threshold
        history.append(raw_complete)
        switch_complete = (
            len(history) == args.window_size
            and sum(history) >= args.required_positive
        )
        color = (40, 200, 40) if switch_complete else (40, 40, 220)
        label = "TRUE" if switch_complete else "FALSE"
        cv2.rectangle(current_bgr, (16, 16), (520, 112), (0, 0, 0), thickness=-1)
        cv2.putText(
            current_bgr,
            f"TASK COMPLETE: {label}",
            (30, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            current_bgr,
            f"p={probability:.3f} threshold={threshold:.3f} raw={raw_complete}",
            (30, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        writer.write(current_bgr)
        frame_index += 1

        ok, current_bgr = capture.read()
        if not ok:
            break

    capture.release()
    writer.release()
    print(
        f"wrote {frame_index} annotated frames to {args.output} "
        f"(fps={fps:.3f}, task={args.text!r})"
    )


if __name__ == "__main__":
    main()
