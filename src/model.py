import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=38, freeze_backbone=True):
    model = models.resnet50(weights="IMAGENET1K_V2")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    return model