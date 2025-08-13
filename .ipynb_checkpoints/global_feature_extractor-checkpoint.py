# global_feature_extractor.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# -------------------------------
# Install VMamba (run once)
# -------------------------------
# !git clone https://github.com/MzeroMiko/VMamba.git
# !pip install -e VMamba

from vmamba import VSSM  # Import from the VMamba repo

# -------------------------------
# Configuration
# -------------------------------
IMAGE_ROOT = "."  # Root directory containing images
CSV_PATH = "./text/des.csv"
OUTPUT_PATH = "./features/global_features.pt"  # # "./features/OIQA/global_features.pt" "./features/CVIQ/global_features.pt"
IMG_SIZE = (512, 1024)  # Resize ERP images for efficiency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# -------------------------------
# Load image list
# -------------------------------
df = pd.read_csv(CSV_PATH)
image_names = df["image_name"].tolist()
print(f"Found {len(image_names)} images.")

# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize ERP: e.g., 1024x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# Load VMamba Model with Hook for Intermediate Features
# -------------------------------
class VMambaFeatureExtractor(nn.Module):
    def __init__(self, model_name="vmamba_tiny", pretrained=True):
        super().__init__()
        # Load pre-trained VMamba model
        self.model = VSSM(depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], pretrained=pretrained)
        self.model.eval()

        # We'll capture outputs from stages 2, 3, 4 (index 1, 2, 3)
        self.features = []

        def make_hook(stage_idx):
            def hook(module, input, output):
                self.features.append(output)
            return hook

        # Register forward hooks on the last layer of each stage
        self.model.layers[1].blocks[-1].register_forward_hook(make_hook(1))
        self.model.layers[2].blocks[-1].register_forward_hook(make_hook(2))
        self.model.layers[3].blocks[-1].register_forward_hook(make_hook(3))

    def forward(self, x):
        self.features = []
        _ = self.model(x)  # Forward pass triggers hooks
        # Return f2, f3, f4
        return self.features  # List of [f2, f3, f4]

# -------------------------------
# Initialize model
# -------------------------------
model = VMambaFeatureExtractor(pretrained=True)
model = model.to(DEVICE)
model.model.forward = lambda x: model.model.forward_features(x)  # Only feature extraction
model.eval()

# -------------------------------
# Extract global features
# -------------------------------
all_features = {
    "f2": [],
    "f3": [],
    "f4": [],
}

print("Starting global feature extraction...")

with torch.no_grad():
    for img_name in image_names:
        img_path = os.path.join(IMAGE_ROOT, img_name.strip())
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

        f2, f3, f4 = model(img_tensor)  # Each is (1, C, H', W')

        all_features["f2"].append(f2.cpu())
        all_features["f3"].append(f3.cpu())
        all_features["f4"].append(f4.cpu())

        print(f"Processed {img_name}: f2{f2.shape}, f3{f3.shape}, f4{f4.shape}")

# Stack all
f2_tensor = torch.cat(all_features["f2"], dim=0)  # [N, 192, H2, W2]
f3_tensor = torch.cat(all_features["f3"], dim=0)  # [N, 384, H3, W3]
f4_tensor = torch.cat(all_features["f4"], dim=0)  # [N, 768, H4, W4]

print(f"Final shapes: f2 {f2_tensor.shape}, f3 {f3_tensor.shape}, f4 {f4_tensor.shape}")

# -------------------------------
# Save features
# -------------------------------
torch.save({
    "image_names": image_names,
    "f2": f2_tensor,
    "f3": f3_tensor,
    "f4": f4_tensor,
    "model": "MzeroMiko/VMamba",
    "variant": "vmamba_tiny",
    "depths": [2, 2, 9, 2],
    "dims": [96, 192, 384, 768],
}, OUTPUT_PATH)

print(f"Global features saved to {OUTPUT_PATH}")
