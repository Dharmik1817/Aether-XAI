"""
Aether-XAI — model.py
SIH1521 (ISRO) — Explainable AI for heavy rain event prediction

WHAT THIS FILE DOES
--------------------
1. Loads a pretrained ResNet-50 backbone and REPLACES its final layer with a
   small binary head (rain / no_rain) — this is the fix for the "it's just
   ImageNet" problem. Grad-CAM is only meaningful once gradients flow from a
   rain-relevant class, not an ImageNet class.
2. Fine-tunes ONLY that new head (backbone frozen) on your labeled images —
   fast enough to run on a free Colab GPU in well under an hour for a
   demo-scale dataset (even 100-300 images per class is enough to produce a
   convincing, semantically real Grad-CAM heatmap).
3. Implements Grad-CAM against the trained "rain" class on the last conv
   block (layer4) — so the heatmap highlights storm-relevant regions.
4. Implements the two concrete, judge-defensible gauges:
      - Signal Noise Gauge  -> Laplacian Variance (OpenCV)
      - Uncertainty Gauge   -> Shannon Entropy of the softmax output
5. Ships a --demo mode that auto-generates synthetic "clear" and "noisy"
   sample images if you don't have a labeled dataset yet, so the whole
   pipeline (train -> predict -> Grad-CAM -> gauges) runs end-to-end today,
   even before your Data Lead finds real INSAT-3D TIR/WV samples.

FOLDER LAYOUT EXPECTED FOR REAL DATA
-------------------------------------
data/
  train/
    rain/       *.png or *.jpg   (heavy-rain-labeled TIR/WV crops)
    no_rain/    *.png or *.jpg
  val/
    rain/
    no_rain/

USAGE
-----
    # 1) One-time: generate synthetic demo data + train a quick head on it
    python model.py --demo --train

    # 2) Once your Data Lead has a real dataset in ./data/, train on it:
    python model.py --data_dir ./data --train

    # 3) Run inference + Grad-CAM + gauges on a single image
    python model.py --predict path/to/image.png

Requires: torch, torchvision, opencv-python, numpy, pillow, matplotlib
    pip install torch torchvision opencv-python numpy pillow matplotlib
"""

import os
import argparse
import random

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["no_rain", "rain"]  # index 0 / index 1 — keep this order everywhere
IMG_SIZE = 224
CHECKPOINT_PATH = "aether_xai_head.pt"


# ---------------------------------------------------------------------------
# 1. MODEL DEFINITION — ResNet-50 backbone (frozen) + trainable binary head
# ---------------------------------------------------------------------------
class AetherXAIModel(nn.Module):
    """
    Wraps torchvision's ResNet-50. We freeze every backbone layer and replace
    the 1000-way ImageNet classifier with a fresh 2-way (rain / no_rain)
    linear head. Only the head's weights get trained — this is standard
    transfer learning, and it's what makes Grad-CAM's output mean something
    for THIS task instead of ImageNet's.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # IMPORTANT: we do NOT set requires_grad=False on backbone params here.
        # "Freezing" for training purposes is done later by only handing the
        # head's parameters to the optimizer (see train_head). If we set
        # requires_grad=False here instead, autograd stops building a graph
        # through layer4 entirely, and Grad-CAM's backward hook silently gets
        # no gradient (self.gradients stays None) -> crashes at inference.
        # Keeping requires_grad=True on the backbone costs a bit of extra
        # backward-pass compute but is what makes Grad-CAM work at all.

        in_features = backbone.fc.in_features  # 2048 for ResNet-50
        backbone.fc = nn.Linear(in_features, len(CLASS_NAMES))  # trainable head
        self.backbone = backbone

        # Keep a handle on layer4 — this is where Grad-CAM will hook in.
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# 2. DATASET — real folder-based dataset, or synthetic demo generator
# ---------------------------------------------------------------------------
class FolderRainDataset(Dataset):
    """Expects data_dir/rain/*.png and data_dir/no_rain/*.png"""

    def __init__(self, data_dir, transform):
        self.samples = []
        self.transform = transform
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_synthetic_demo_dataset(root="demo_data", n_per_class=60):
    """
    Generates fake grayscale-ish 'satellite-like' images so the full
    pipeline can be trained and demoed TODAY, before real INSAT data is
    ready. 'rain' images get bright, high-contrast blob clusters (simulating
    dense convective storm cells / cold cloud tops). 'no_rain' images get
    flat, low-contrast noise. This is ONLY for pipeline testing and demo
    rehearsal — say so plainly if a judge asks what data trained the model.
    """
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(42)

    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            out_dir = os.path.join(root, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            n = n_per_class if split == "train" else max(10, n_per_class // 5)

            for i in range(n):
                canvas = rng.normal(loc=90, scale=15, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)

                if class_name == "rain":
                    # Simulate a dense, bright, cold cloud-top cluster (a storm cell)
                    n_blobs = rng.integers(2, 5)
                    for _ in range(n_blobs):
                        cx, cy = rng.integers(40, IMG_SIZE - 40, size=2)
                        r = rng.integers(20, 45)
                        y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
                        canvas[mask] += rng.uniform(90, 140)
                else:
                    # Flat field with mild random texture, no coherent structure
                    canvas += rng.normal(loc=0, scale=8, size=(IMG_SIZE, IMG_SIZE))

                canvas = np.clip(canvas, 0, 255).astype(np.uint8)
                img = Image.fromarray(canvas).convert("RGB")
                img.save(os.path.join(out_dir, f"{class_name}_{i:03d}.png"))

    print(f"Synthetic demo dataset created at ./{root}/ "
          f"({n_per_class} train + {max(10, n_per_class // 5)} val images per class).")
    return os.path.join(root, "train"), os.path.join(root, "val")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# 3. TRAINING — only the head's params get updated
# ---------------------------------------------------------------------------
def train_head(model, train_dir, val_dir, epochs=6, batch_size=16, lr=1e-3):
    transform = get_transforms()
    train_ds = FolderRainDataset(train_dir, transform)
    val_ds = FolderRainDataset(val_dir, transform)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No images found in {train_dir}/rain or {train_dir}/no_rain. "
            "Populate that folder or run with --demo first."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Explicitly train ONLY the head's parameters. We deliberately do NOT use
    # requires_grad filtering here (see note in AetherXAIModel.__init__) —
    # every backbone param still has requires_grad=True so Grad-CAM can work
    # later, but only backbone.fc's weights are ever handed to the optimizer,
    # so only the head actually updates during training.
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / max(total, 1)

        # quick val pass
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)
        val_acc = v_correct / max(v_total, 1)

        print(f"Epoch {epoch+1}/{epochs} | loss {total_loss/max(total,1):.4f} "
              f"| train_acc {train_acc:.2%} | val_acc {val_acc:.2%}")

    torch.save(model.backbone.fc.state_dict(), CHECKPOINT_PATH)
    print(f"Saved trained head to {CHECKPOINT_PATH}")


# ---------------------------------------------------------------------------
# 4. GRAD-CAM — hooked to layer4, gradients taken w.r.t. the trained "rain" class
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: AetherXAIModel):
        self.model = model
        self.gradients = None
        self.activations = None
        model.target_layer.register_forward_hook(self._save_activation)
        model.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=1):
        """class_idx=1 -> 'rain'. Returns a normalized (H,W) heatmap in [0,1]."""
        self.model.eval()
        output = self.model(input_tensor)  # forward pass populates self.activations
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward()  # backward pass populates self.gradients

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # only positive influence on the "rain" class
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, torch.softmax(output, dim=1).detach().cpu().numpy()[0]


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha=0.45):
    """Blends the Grad-CAM heatmap onto the original image for display in the UI."""
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# 5. THE TWO GAUGES — concrete, explainable, cheap to compute
# ---------------------------------------------------------------------------
def signal_noise_score(pil_img: Image.Image) -> float:
    """
    Laplacian Variance: Var(∇²I).
    High value  -> crisp, well-defined cloud edges (clean signal).
    Low value   -> blur / scatter / dropped data (degraded satellite feed).
    """
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def prediction_uncertainty(softmax_probs: np.ndarray) -> float:
    """
    Shannon Entropy: H(P) = -sum(p_i * log(p_i)).
    ~0     -> model is confident (probability mass on one class)
    ~log(2)-> model is maximally confused between rain / no_rain (~0.693 for 2 classes)
    """
    eps = 1e-12  # avoid log(0)
    return float(-np.sum(softmax_probs * np.log(softmax_probs + eps)))


def interpret_gauges(noise_score: float, entropy: float,
                      noise_threshold=15.0, entropy_threshold=0.5):
    """
    Turns the two raw numbers into the plain-language advisory your Policy
    Lead's SDMA text generator can consume. Thresholds here are placeholders
    tuned for the synthetic demo data — recalibrate once you have real,
    labeled INSAT samples (e.g. by checking the score distribution on your
    val set and picking a threshold that separates known-clean vs
    known-noisy examples).
    """
    flags = []
    if noise_score < noise_threshold:
        flags.append("HIGH SIGNAL NOISE — satellite feed may be degraded. "
                      "Recommend human meteorologist verification.")
    if entropy > entropy_threshold:
        flags.append("HIGH PREDICTION UNCERTAINTY — model is not confident. "
                      "Do not auto-trigger SDMA protocol without review.")
    if not flags:
        flags.append("Signal clean, prediction confident — safe to surface to SDMA dashboard.")
    return flags


# ---------------------------------------------------------------------------
# 6. END-TO-END PREDICT — what app.py (Streamlit) should call
# ---------------------------------------------------------------------------
def predict_image(model, image_path):
    transform = get_transforms()
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    cam_engine = GradCAM(model)
    cam, probs = cam_engine.generate(input_tensor, class_idx=CLASS_NAMES.index("rain"))
    overlay = overlay_heatmap(pil_img, cam)

    noise = signal_noise_score(pil_img)
    entropy = prediction_uncertainty(probs)
    advisories = interpret_gauges(noise, entropy)

    result = {
        "prediction": CLASS_NAMES[int(np.argmax(probs))],
        "rain_probability": float(probs[CLASS_NAMES.index("rain")]),
        "no_rain_probability": float(probs[CLASS_NAMES.index("no_rain")]),
        "signal_noise_score": noise,
        "prediction_entropy": entropy,
        "advisories": advisories,
        "heatmap_overlay": overlay,  # PIL Image — hand this straight to st.image()
    }
    return result


def load_model_with_head(checkpoint_path=CHECKPOINT_PATH):
    model = AetherXAIModel(freeze_backbone=True)
    if os.path.exists(checkpoint_path):
        model.backbone.fc.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded trained head from {checkpoint_path}")
    else:
        print("WARNING: no trained head checkpoint found — predictions will be meaningless "
              "until you run --train (or --demo --train) at least once.")
    model.to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether-XAI model pipeline")
    parser.add_argument("--demo", action="store_true",
                         help="Generate a synthetic demo dataset in ./demo_data")
    parser.add_argument("--data_dir", type=str, default="demo_data",
                         help="Root folder containing train/ and val/ subfolders")
    parser.add_argument("--train", action="store_true", help="Fine-tune the classifier head")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--predict", type=str, default=None,
                         help="Path to a single image to run inference on")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    train_dir, val_dir = None, None
    if args.demo:
        train_dir, val_dir = build_synthetic_demo_dataset(root=args.data_dir)
    elif args.train:
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

    if args.train:
        model = AetherXAIModel(freeze_backbone=True)
        train_head(model, train_dir, val_dir, epochs=args.epochs)

    if args.predict:
        model = load_model_with_head()
        result = predict_image(model, args.predict)
        print("\n--- Aether-XAI Prediction ---")
        print(f"Prediction:            {result['prediction']}")
        print(f"Rain probability:      {result['rain_probability']:.2%}")
        print(f"Signal noise score:    {result['signal_noise_score']:.2f}")
        print(f"Prediction entropy:    {result['prediction_entropy']:.3f}")
        print("Advisories:")
        for a in result["advisories"]:
            print(f"  - {a}")
        out_path = "gradcam_overlay.png"
        result["heatmap_overlay"].save(out_path)
        print(f"Grad-CAM overlay saved to {out_path}")
