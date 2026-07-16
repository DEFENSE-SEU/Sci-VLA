#!/usr/bin/env python
"""Train the shared vision-language task-completion switch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from manifest_dataset import CompletionCollator, CompletionManifestDataset
from model import CompletionSwitchModel, load_checkpoint, save_checkpoint


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = labels >= 0.5
    tp = int(np.logical_and(predictions, truth).sum())
    fp = int(np.logical_and(predictions, ~truth).sum())
    tn = int(np.logical_and(~predictions, ~truth).sum())
    fn = int(np.logical_and(~predictions, truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_precision: float,
) -> tuple[float, dict[str, float]]:
    candidates = [
        binary_metrics(labels, probabilities, float(threshold))
        for threshold in np.linspace(0.05, 0.95, 91)
    ]
    feasible = [item for item in candidates if item["precision"] >= min_precision and item["tp"] > 0]
    if feasible:
        best = max(feasible, key=lambda item: (item["recall"], item["f1"], item["threshold"]))
    else:
        best = max(candidates, key=lambda item: (item["f1"], item["precision"]))
    return best["threshold"], best


@torch.inference_mode()
def evaluate(
    model: CompletionSwitchModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = _move(batch, device)
        targets = batch.pop("labels")
        logits = model(**batch)
        losses.append(float(criterion(logits, targets).item()))
        labels.append(targets.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name)
    collator = CompletionCollator(processor)
    train_dataset = CompletionManifestDataset(args.manifest, "train")
    val_dataset = CompletionManifestDataset(args.manifest, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    train_labels = np.asarray(
        [float(record["task_is_complete"]) for record in train_dataset.records],
        dtype=np.float32,
    )
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Training split must contain both labels; positives={positives}, negatives={negatives}"
        )
    pos_weight = torch.tensor([negatives / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = CompletionSwitchModel(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_f1 = -1.0
    checkpoint_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.jsonl"

    with history_path.open("w") as history_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch:02d}"):
                batch = _move(batch, device)
                targets = batch.pop("labels")
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

            val_loss, val_labels, val_probabilities = evaluate(model, val_loader, device, criterion)
            threshold, metrics = select_threshold(
                val_labels,
                val_probabilities,
                min_precision=args.min_precision,
            )
            record = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                "val_loss": val_loss,
                **metrics,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(json.dumps(record, indent=2))

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    epoch=epoch,
                    threshold=threshold,
                    metrics=record,
                )

    best_model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    try:
        test_dataset = CompletionManifestDataset(args.manifest, "test")
    except ValueError:
        print("No test split records; skipping final test evaluation.")
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    test_loss, test_labels, test_probabilities = evaluate(best_model, test_loader, device, criterion)
    test_metrics = binary_metrics(
        test_labels,
        test_probabilities,
        float(checkpoint["threshold"]),
    )
    result = {"test_loss": test_loss, **test_metrics}
    with (args.output_dir / "test_metrics.json").open("w") as file:
        json.dump(result, file, indent=2)
    print("test:", json.dumps(result, indent=2))
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
