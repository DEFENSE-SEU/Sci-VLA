from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel


class CompletionSwitchModel(nn.Module):
    """Frozen CLIP encoders with a trainable goal-conditioned binary head."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.clip = CLIPModel.from_pretrained(model_name)
        embedding_dim = int(self.clip.config.projection_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim * 6),
            nn.Linear(embedding_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.clip.parameters()):
            self.clip.eval()
        return self

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(features, dim=-1)

    def classify_features(
        self,
        current: torch.Tensor,
        initial: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        delta = current - initial
        fusion = torch.cat(
            [
                current,
                initial,
                delta,
                text,
                current * text,
                delta * text,
            ],
            dim=-1,
        )
        return self.classifier(fusion).squeeze(-1)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        initial_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = self.encode_image(pixel_values)
        initial = self.encode_image(initial_pixel_values)
        text = self.encode_text(input_ids, attention_mask)
        return self.classify_features(current, initial, text)


def save_checkpoint(
    path: Path,
    model: CompletionSwitchModel,
    *,
    epoch: int,
    threshold: float,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": model.model_name,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "classifier": model.classifier.state_dict(),
            "epoch": int(epoch),
            "threshold": float(threshold),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CompletionSwitchModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    model = CompletionSwitchModel(
        model_name=checkpoint["model_name"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        freeze_backbone=True,
    )
    model.classifier.load_state_dict(checkpoint["classifier"])
    model.to(device)
    model.eval()
    return model, checkpoint

