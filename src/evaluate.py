import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from pathlib import Path
from model import build_model
from dataset import PlantDiseaseDataset, get_test_transforms
from utils import load_class_names, load_checkpoint, get_device, save_metrics, print_metrics


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = F.softmax(outputs, dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(cm, class_names, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, linecolor="#e0e0e0", ax=ax)
    ax.set_title("Confusion Matrix - PlantVillage Test Set", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    device = get_device()
    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)

    transform = get_test_transforms()
    dataset = PlantDiseaseDataset(args.test_csv, args.img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=num_classes)
    load_checkpoint(model, args.checkpoint, device=device)

    preds, labels = run_inference(model, loader, device)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    metrics = {
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_samples": int(len(labels)),
        "num_classes": num_classes,
    }
    print_metrics(metrics)
    save_metrics(metrics, "outputs/metrics.json")

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    rp = Path("outputs/classification_report.txt")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(report)

    cm = confusion_matrix(labels, preds)
    save_confusion_matrix(cm, class_names, "outputs/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="outputs/best_model.pth")
    parser.add_argument("--test_csv",    default="data/splits/test.csv")
    parser.add_argument("--img_dir",     default="data/images")
    parser.add_argument("--class_names", default="class_names.txt")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())