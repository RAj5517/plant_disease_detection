import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import build_model
from utils import load_class_names, load_checkpoint, get_device, format_class_name


class GradCAM:
    def __init__(self, model):
        self.model = model
        self._features = None
        self._grads = None
        target = self.model.layer4[-1]
        target.register_forward_hook(lambda m, i, o: setattr(self, "_features", o))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def generate(self, x, cls):
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self._features).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)


def to_rgb(cam):
    import matplotlib.cm as mcm
    return (mcm.get_cmap("RdYlGn")(cam)[:, :, :3] * 255).astype(np.uint8)


def overlay(img, cam, a=0.45):
    img224 = cv2.resize(img, (224, 224))
    return (a * to_rgb(cam) + (1 - a) * img224).astype(np.uint8)


def save_figure(orig, cam, label, conf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Grad-CAM | {label} ({conf*100:.1f}%)", fontsize=13, fontweight="bold")
    axes[0].imshow(cv2.resize(orig, (224, 224))); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(to_rgb(cam));                  axes[1].set_title("Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay(orig, cam));           axes[2].set_title("Overlay");  axes[2].axis("off")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def main(args):
    device = get_device()
    class_names = load_class_names(args.class_names)
    model = build_model(num_classes=len(class_names))
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    gcam = GradCAM(model)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    img_np = np.array(Image.open(args.image).convert("RGB"))
    x = transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    top_probs, top_idxs = torch.topk(probs, 3)
    cls = top_idxs[0][0].item()
    conf = top_probs[0][0].item()

    with torch.enable_grad():
        cam = gcam.generate(x.clone(), cls)

    label = format_class_name(class_names[cls])
    print(f"Prediction: {label} ({conf*100:.2f}%)")

    out = args.output or f"outputs/gradcam_{Path(args.image).stem}.png"
    save_figure(img_np, cam, label, conf, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True)
    parser.add_argument("--checkpoint",  default="outputs/best_model.pth")
    parser.add_argument("--class_names", default="class_names.txt")
    parser.add_argument("--output",      default=None)
    main(parser.parse_args())