from typing import Optional
import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_backbone_epochs: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_model(
    backbone: str = "efficientnet_b4",
    pretrained: bool = True,
    dropout: float = 0.4,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> DeepfakeDetector:
    model = DeepfakeDetector(backbone=backbone, pretrained=pretrained, dropout=dropout)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])

    return model.to(device)
