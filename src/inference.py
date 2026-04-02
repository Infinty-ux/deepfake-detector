import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model

TTA_TRANSFORMS = [
    A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224),
               A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
    A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224), A.HorizontalFlip(p=1.0),
               A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
]

BASE_TRANSFORM = TTA_TRANSFORMS[0]


class DeepfakeInference:
    def __init__(
        self,
        checkpoint: str,
        backbone: str = "efficientnet_b4",
        device: str = "cpu",
        threshold: float = 0.5,
        use_tta: bool = False,
    ):
        self.device = device
        self.threshold = threshold
        self.use_tta = use_tta
        self.model = build_model(backbone=backbone, pretrained=False, checkpoint_path=checkpoint, device=device)
        self.model.eval()

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        return np.array(Image.open(str(path)).convert("RGB"))

    @torch.no_grad()
    def _forward(self, img: np.ndarray) -> float:
        if self.use_tta:
            probs = []
            for t in TTA_TRANSFORMS:
                tensor = t(image=img)["image"].unsqueeze(0).to(self.device)
                probs.append(torch.sigmoid(self.model(tensor)).item())
            return float(np.mean(probs))
        else:
            tensor = BASE_TRANSFORM(image=img)["image"].unsqueeze(0).to(self.device)
            return torch.sigmoid(self.model(tensor)).item()

    def predict_image(self, path: Union[str, Path]) -> Dict:
        img = self._load_image(path)
        prob = self._forward(img)
        return {
            "path": str(path),
            "fake_probability": round(prob, 4),
            "real_probability": round(1 - prob, 4),
            "prediction": "fake" if prob >= self.threshold else "real",
            "confidence": round(max(prob, 1 - prob), 4),
        }

    def predict_batch(self, paths: List[Union[str, Path]], batch_size: int = 16) -> List[Dict]:
        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i: i + batch_size]
            for p in batch_paths:
                try:
                    results.append(self.predict_image(p))
                except Exception as e:
                    results.append({"path": str(p), "error": str(e)})
        return results

    def predict_directory(self, directory: Union[str, Path], exts=None) -> List[Dict]:
        exts = exts or {".jpg", ".jpeg", ".png", ".webp"}
        paths = [f for f in Path(directory).rglob("*") if f.suffix.lower() in exts]
        return self.predict_batch(paths)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="Image file or directory")
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    inf = DeepfakeInference(args.checkpoint, backbone=args.backbone,
                            device=args.device, threshold=args.threshold, use_tta=args.tta)

    path = Path(args.input)
    if path.is_dir():
        results = inf.predict_directory(path)
    else:
        results = [inf.predict_image(path)]

    for r in results:
        print(json.dumps(r))
