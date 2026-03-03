"""
utils.py - Shared utilities for Plant Disease Detection
"""

import random
import json
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels, num_classes: int, device=torch.device("cpu")) -> torch.Tensor:
    """
    Inverse-frequency class weights matching the training formula:
        weight_i = N / (C * n_i)

    Parameters
    ----------
    labels      : flat list / array of integer class labels
    num_classes : total number of classes (38 for PlantVillage)
    device      : target device for the returned tensor

    Returns
    -------
    torch.Tensor of shape (num_classes,)
    """
    labels = np.array(labels)
    N = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        n_i = int(np.sum(labels == i))
        weights[i] = N / (num_classes * n_i) if n_i > 0 else 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path) -> None:
    """Save a full training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        path,
    )
    print(f"[Checkpoint] Saved -> {path}  (epoch {epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(model, path, optimizer=None, device=torch.device("cpu")) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).
    Handles both bare state_dicts and full checkpoint dicts.
    Returns the full checkpoint dict.
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded <- {path}")
    return ckpt


def load_class_names(path: str = "class_names.txt") -> list:
    """Load the ordered list of class names from a plain-text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def format_class_name(raw: str) -> str:
    """
    Convert an internal class name to a human-readable label.
    Example: 'Tomato___Early_blight' -> 'Tomato - Early Blight'
    """
    parts = raw.replace("___", "|||").split("|||")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip().title()
        return f"{plant} - {disease}"
    return raw.replace("_", " ").title()


def save_metrics(metrics: dict, path: str = "outputs/metrics.json") -> None:
    """Persist a metrics dict as a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved -> {path}")


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict."""
    col = 28
    print("\n" + "-" * 45)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<{col}} {v:.4f}")
        else:
            print(f"  {k:<{col}} {v}")
    print("-" * 45 + "\n")


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    return device