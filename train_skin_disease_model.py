from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from skinscan_api.config import get_settings  # noqa: E402


def build_model(architecture: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.00)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def split_indices(count: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    split_at = max(1, int(count * train_ratio))
    split_at = min(split_at, count - 1) if count > 1 else count
    return indices[:split_at], indices[split_at:]


def make_dataloaders(data_dir: Path, image_size: int, batch_size: int, train_ratio: float, seed: int) -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform, eval_transform = build_transforms(image_size)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        class_names = train_dataset.classes
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, class_names

    class_dirs = [child for child in data_dir.iterdir() if child.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"No image class folders were found in {data_dir}. Add class subfolders like {data_dir / 'melanoma'} or a train/val split before training."
        )

    base_dataset = datasets.ImageFolder(data_dir)
    train_indices, val_indices = split_indices(len(base_dataset), train_ratio, seed)

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    class_names = base_dataset.classes

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer | None, device: torch.device) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_seen += batch_size

    average_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    return average_loss, accuracy


def save_checkpoint(output_path: Path, model: nn.Module, class_names: list[str], architecture: str, image_size: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "class_names": class_names,
            "image_size": image_size,
            "model_state_dict": model.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        output_path,
    )


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train a skin disease classifier checkpoint for SkinScan-AI.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "frontend" / "Dataset", help="Dataset root containing class folders or train/val splits.")
    parser.add_argument("--output", type=Path, default=Path(settings.model_path), help="Checkpoint path to write.")
    parser.add_argument("--architecture", choices=["resnet18", "efficientnet_b0"], default="resnet18", help="Backbone architecture to train.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Start from torchvision default weights when available.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_loader, val_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if not class_names:
        raise ValueError(f"No class folders were found in {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_accuracy = 0.0
    best_train_metrics = {"loss": 0.0, "accuracy": 0.0}
    best_val_metrics = {"loss": 0.0, "accuracy": 0.0}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_train_metrics = {"loss": train_loss, "accuracy": train_accuracy}
            best_val_metrics = {"loss": val_loss, "accuracy": val_accuracy}
            save_checkpoint(args.output, model, class_names, args.architecture, args.image_size, best_train_metrics, best_val_metrics)

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "class_names": class_names,
                "image_size": args.image_size,
                "best_train_metrics": best_train_metrics,
                "best_val_metrics": best_val_metrics,
                "checkpoint_path": str(args.output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved checkpoint to {args.output}")
    print(f"Saved metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())