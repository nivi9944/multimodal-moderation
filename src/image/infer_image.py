"""EfficientNet-B0 image scorer: P(harmful | image)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import MODELS

IMG_SIZE = 256
# ImageNet normalisation, matching the pretrained timm weights
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(train: bool):
    from torchvision import transforms as T

    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model(pretrained: bool = True):
    """EfficientNet-B0 with a single logit output (sigmoid gives P(harmful))."""
    import timm

    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)


class ImageScorer:
    def __init__(self, weights: Path = MODELS / "effnet_b0_memes.pt", device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()
        self.tf = build_transforms(train=False)

    def predict(self, images: list, batch: int = 32) -> np.ndarray:
        """images: list of PIL images. Returns P(harmful) per image."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch):
                x = self.torch.stack([self.tf(im.convert("RGB")) for im in images[i : i + batch]]).to(self.device)
                out.append(self.torch.sigmoid(self.model(x).squeeze(-1)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
