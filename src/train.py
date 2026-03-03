import sys
sys.path.append("src")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from model import build_resnet50
from dataset import PlantDiseaseDataset
from augmentation import get_train_transforms, get_val_transforms


# ============================
# PATH CONFIGURATION
# ============================

SPLIT_DIR = "data/splits"
CHECKPOINT_DIR = "/content/drive/MyDrive/plant_assets/checkpoints"
BEST_MODEL_PATH = "/content/drive/MyDrive/plant_assets/best_model.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================
# CLASS WEIGHT CALCULATION
# ============================

def compute_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df["label_index"].value_counts().sort_index()
    total = len(df)
    num_classes = len(counts)

    weights = total / (num_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float)


# ============================
# TRAIN FUNCTION
# ============================

def train():

    train_dataset = PlantDiseaseDataset(
        os.path.join(SPLIT_DIR, "train.csv"),
        transform=get_train_transforms()
    )

    val_dataset = PlantDiseaseDataset(
        os.path.join(SPLIT_DIR, "val.csv"),
        transform=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ----------------------------
    # MODEL
    # ----------------------------
    model = build_resnet50()
    model.to(DEVICE)

    class_weights = compute_class_weights(
        os.path.join(SPLIT_DIR, "train.csv")
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # =============================
    # PHASE 1 — Train Head Only
    # =============================
    print("\n===== PHASE 1: Training classifier head only =====")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    best_val_acc = 0.0
    patience = 7
    epochs_no_improve = 0

    for epoch in range(5):

        print(f"\nPhase 1 - Epoch {epoch+1}/5")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, f"{CHECKPOINT_DIR}/phase1_epoch_{epoch+1}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered in Phase 1.")
            break

    # =============================
    # PHASE 2 — Fine Tune layer4
    # =============================
    print("\n===== PHASE 2: Fine-tuning layer4 + head =====")

    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    epochs_no_improve = 0

    for epoch in range(25):

        print(f"\nPhase 2 - Epoch {epoch+1}/25")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, f"{CHECKPOINT_DIR}/phase2_epoch_{epoch+1}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Saved best model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered in Phase 2.")
            break

    print("Training complete.")


if __name__ == "__main__":
    train()