import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from utils import load_class_names, load_checkpoint, get_device, format_class_name


MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def predict_single(model, img_np, transform, class_names, device, top_k=3):
    x = transform(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    top_probs, top_idxs = torch.topk(probs, top_k)
    results = []
    for i in range(top_k):
        results.append({
            "rank": i + 1,
            "class_raw": class_names[top_idxs[0][i].item()],
            "class_label": format_class_name(class_names[top_idxs[0][i].item()]),
            "confidence": round(top_probs[0][i].item(), 4),
            "confidence_pct": round(top_probs[0][i].item() * 100, 2),
        })
    return results


def predict_folder(model, folder, transform, class_names, device, top_k=3):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in extensions]
    print(f"Found {len(image_paths)} images in {folder}")

    all_results = {}
    for idx, img_path in enumerate(image_paths, 1):
        try:
            img_np = load_image(img_path)
            preds = predict_single(model, img_np, transform, class_names, device, top_k)
            all_results[str(img_path)] = preds
            top = preds[0]
            print(f"[{idx:>4}/{len(image_paths)}] {img_path.name:<40} "
                  f"{top['class_label']} ({top['confidence_pct']:.1f}%)")
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            all_results[str(img_path)] = {"error": str(e)}

    return all_results


def print_predictions(preds, image_path=None):
    if image_path:
        print(f"\nImage : {image_path}")
    print("-" * 55)
    print(f"  {'Rank':<6} {'Label':<35} {'Confidence':>10}")
    print("-" * 55)
    for p in preds:
        print(f"  {p['rank']:<6} {p['class_label']:<35} {p['confidence_pct']:>9.2f}%")
    print("-" * 55)


def main(args):
    device = get_device()
    class_names = load_class_names(args.class_names)
    model = build_model(num_classes=len(class_names))
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    transform = get_transform()

    if args.image:
        img_np = load_image(args.image)
        preds = predict_single(model, img_np, transform, class_names, device, top_k=args.top_k)
        print_predictions(preds, image_path=args.image)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({str(args.image): preds}, f, indent=2)
            print(f"\nSaved -> {args.output}")

    elif args.folder:
        results = predict_folder(model, args.folder, transform, class_names, device, top_k=args.top_k)
        out_path = args.output or "outputs/batch_predictions.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch results saved -> {out_path}")

    else:
        print("Provide --image or --folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       default=None,  help="Path to a single image")
    parser.add_argument("--folder",      default=None,  help="Path to a folder of images")
    parser.add_argument("--checkpoint",  default="outputs/best_model.pth")
    parser.add_argument("--class_names", default="class_names.txt")
    parser.add_argument("--top_k",       type=int, default=3)
    parser.add_argument("--output",      default=None,  help="Save predictions as JSON")
    main(parser.parse_args())