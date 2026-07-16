from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class CompletionManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open() as file:
            records = [json.loads(line) for line in file if line.strip()]
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as image_file:
            image = image_file.convert("RGB").copy()
        with Image.open(self.root / record["initial_image"]) as image_file:
            initial_image = image_file.convert("RGB").copy()
        return {
            "image": image,
            "initial_image": initial_image,
            "text": record["task_description"],
            "label": float(record["task_is_complete"]),
        }


class CompletionCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        current = self.processor.image_processor(
            images=[item["image"] for item in batch],
            return_tensors="pt",
        )["pixel_values"]
        initial = self.processor.image_processor(
            images=[item["initial_image"] for item in batch],
            return_tensors="pt",
        )["pixel_values"]
        text = self.processor.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": current,
            "initial_pixel_values": initial,
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        }

