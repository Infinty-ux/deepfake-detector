# Deepfake Detector

A production-ready CNN-based classifier for detecting AI-generated / manipulated images, built on **EfficientNet-B4** with a custom classification head, trained with **mixed-precision** (AMP) and **GradCAM** explainability.

## Architecture

```
Input (224×224 RGB)
        │
EfficientNet-B4 backbone (timm)
  └── Pretrained ImageNet features
        │ (2048-dim feature vector)
        │
Custom head:
  Linear(2048→512) → BN → GELU → Dropout(0.4)
  Linear(512→128)  → BN → GELU → Dropout(0.2)
  Linear(128→1)    → Sigmoid
        │
   Fake probability [0, 1]
```

**Training features:**
- Mixed precision (torch.cuda.amp) for 2× faster GPU training
- Weighted random sampler for class imbalance
- Cosine Annealing with Warm Restarts LR schedule
- EarlyStopping on validation AUC
- Test-Time Augmentation (TTA) at inference

## Dataset Structure

```
data/
├── train/
│   ├── real/   ← authentic images
│   └── fake/   ← AI-generated / GAN / diffusion images
└── val/
    ├── real/
    └── fake/
```

Compatible with:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC (Deepfake Detection Challenge)](https://www.kaggle.com/c/deepfake-detection-challenge)
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/trainer.py \
  --data-dir data \
  --output-dir checkpoints \
  --backbone efficientnet_b4 \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-4 \
  --device cuda
```

Best checkpoint saved to `checkpoints/best.pt`. Training history to `checkpoints/history.json`.

## Inference

### Single image

```bash
python src/inference.py --checkpoint checkpoints/best.pt --input image.jpg
```

### Batch / directory

```bash
python src/inference.py --checkpoint checkpoints/best.pt --input images/ --tta
```

Example output:
```json
{"path": "photo.jpg", "fake_probability": 0.9731, "real_probability": 0.0269, "prediction": "fake", "confidence": 0.9731}
```

### GradCAM visualization

```bash
python src/gradcam.py --checkpoint checkpoints/best.pt --image photo.jpg --output gradcam.png
```

Generates a heatmap overlay showing which regions triggered the "fake" classification.

## Project Structure

```
deepfake-detector/
├── src/
│   ├── dataset.py     # Dataset, augmentation pipeline, weighted sampler
│   ├── model.py       # EfficientNet-B4 + custom head
│   ├── trainer.py     # AMP training loop, early stopping, checkpointing
│   ├── evaluator.py   # AUC, F1, precision/recall, threshold analysis
│   ├── inference.py   # Single + batch inference with TTA
│   └── gradcam.py     # GradCAM heatmap generation & overlay
└── requirements.txt
```

## Tech Stack

- **timm** — EfficientNet-B4 pretrained backbone
- **PyTorch AMP** — mixed-precision training (fp16/fp32)
- **albumentations** — fast image augmentation pipeline
- **GradCAM** — gradient-based model explainability
- **scikit-learn** — AUC, F1, confusion matrix metrics
