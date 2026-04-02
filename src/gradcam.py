from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import DeepfakeDetector, build_model


PREPROCESS = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class GradCAM:
    def __init__(self, model: DeepfakeDetector, target_layer_name: Optional[str] = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer = self._find_layer(target_layer_name)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _find_layer(self, name: Optional[str]) -> torch.nn.Module:
        if name is not None:
            for n, m in self.model.named_modules():
                if n == name:
                    return m
        for m in reversed(list(self.model.backbone.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise ValueError("Could not find target layer.")

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        img_tensor = img_tensor.unsqueeze(0).requires_grad_(False)
        img_tensor.requires_grad = True

        logit = self.model(img_tensor)
        prob = torch.sigmoid(logit).item()

        self.model.zero_grad()
        logit.backward()

        gradients = self._gradients.squeeze(0)
        activations = self._activations.squeeze(0)
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, prob

    def overlay(
        self,
        image_path: str,
        output_path: str,
        alpha: float = 0.45,
        device: str = "cpu",
    ):
        img_np = np.array(Image.open(image_path).convert("RGB"))
        tensor = PREPROCESS(image=img_np)["image"].to(device)

        cam, prob = self.generate(tensor)

        img_display = img_np.copy()
        img_display = cv2.resize(img_display, (224, 224))
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlaid = (img_display * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

        label = f"FAKE {prob:.1%}" if prob >= 0.5 else f"REAL {(1-prob):.1%}"
        cv2.putText(overlaid, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2, cv2.LINE_AA)
        Image.fromarray(overlaid).save(output_path)
        return prob


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="gradcam_output.png")
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = build_model(backbone=args.backbone, pretrained=False,
                        checkpoint_path=args.checkpoint, device=args.device)
    gcam = GradCAM(model)
    prob = gcam.overlay(args.image, args.output, device=args.device)
    print(f"Saved GradCAM to {args.output} | fake_prob={prob:.4f}")
