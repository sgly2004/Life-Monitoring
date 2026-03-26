#!/usr/bin/env python3
"""
MentalHealth — Training Script
Trains the Swin Transformer model on FER2013 for mental health detection.
Must be run on a CUDA-enabled machine.

Usage:
    python train_model.py --epochs 50 --batch_size 32 --lr 1e-4
"""
import os
import sys
import json
import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import timm

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MODULE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Emotion labels ──────────────────────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_LABELS_CN = ["生气", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]


# ── Model ─────────────────────────────────────────────────────────────────────
class MentalHealthSwinTransformer(nn.Module):
    """
    Swin Transformer backbone + dropout classifier.
    Matches the architecture from the FER-for-Mental-Health-Detection paper.
    """

    def __init__(self, num_classes=7, dropout=0.6):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_dataloaders(data_root: str, batch_size: int = 32):
    """
    Expects the following directory structure under data_root:
        FER2013_processed/
            train/
                angry/   disgust/   fear/   happy/   sad/   surprise/   neutral/
            val/
                ...
            test/
                ...
    Falls back to flat train/ if subdirectories don't exist.
    """
    fer_root = os.path.join(data_root, "FER2013_processed")
    train_root = os.path.join(fer_root, "train")
    val_root = os.path.join(fer_root, "val")
    test_root = os.path.join(fer_root, "test")

    has_split_dirs = os.path.isdir(train_root) and os.path.isdir(val_root)

    if has_split_dirs:
        train_dataset = datasets.ImageFolder(root=train_root, transform=get_transforms("train"))
        val_dataset = datasets.ImageFolder(root=val_root, transform=get_transforms("val"))
        test_dataset = datasets.ImageFolder(root=test_root, transform=get_transforms("test"))
    else:
        # Flat train/ directory (e.g. after running preprocess_data.py)
        flat_root = os.path.join(fer_root, "train")
        if os.path.isdir(flat_root):
            all_dataset = datasets.ImageFolder(root=flat_root, transform=get_transforms("train"))
            total = len(all_dataset)
            train_size = int(0.8 * total)
            val_size = total - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                all_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            val_dataset.dataset = datasets.ImageFolder(root=flat_root, transform=get_transforms("val"))
            test_dataset = val_dataset
        else:
            raise FileNotFoundError(
                f"FER2013 数据集未找到。请先下载并预处理数据：\n"
                f"  预期路径: {fer_root}/train/\n"
                f"  参考: module/MentalHealth/README.md 中的说明"
            )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[DataLoader] train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_dataset.classes


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    desc = f"Epoch {epoch + 1}/{total_epochs}"

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct.item() / total)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item(), all_preds, all_labels


def compute_class_weights(dataset, device):
    """Compute inverse-frequency class weights for handling imbalanced data."""
    from collections import Counter
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(counter)):
        weight = total / (len(counter) * counter[i])
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train MentalHealth Swin Transformer")
    parser.add_argument("--data_root", type=str, default=MODULE_DIR, help="Root directory containing FER2013_processed/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to resume checkpoint")
    parser.add_argument("--model_save_path", type=str, default="",
                        help="Override model save path (default: model/best_model.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Training on CPU will be very slow.")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.data_root, args.batch_size
    )

    num_classes = len(classes)
    print(f"[Train] Classes: {classes}")
    print(f"[Train] Number of classes: {num_classes}")

    # Save class mapping
    class_map = {str(i): EMOTION_LABELS_CN[i] for i in range(min(num_classes, 7))}
    with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    print(f"[Train] Saved class_mapping.json")

    # Build model
    model = MentalHealthSwinTransformer(num_classes=num_classes, dropout=args.dropout).to(device)

    # Class-weighted loss
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Optionally resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    save_path = args.model_save_path or os.path.join(MODEL_DIR, "best_model.pth")

    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    print(f"[Train] Model will be saved to: {save_path}\n")

    for epoch in range(start_epoch, args.epochs):
        # Layer-wise unfreezing: unfreeze backbone after 10 epochs
        if epoch == 10:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("[Train] Unfrozen backbone layers (epoch 10)")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → New best model saved (val acc: {val_acc:.4f})")

    # Final evaluation on test set
    print("\n[Train] Training complete. Evaluating on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"[Train] Test Accuracy: {test_acc:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print(f"\n[Train] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[Train] Model saved to: {save_path}")


if __name__ == "__main__":
    main()
