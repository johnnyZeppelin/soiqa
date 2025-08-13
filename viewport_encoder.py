# viewport_encoder.py
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import transforms
import torch.nn as nn

# -------------------------------
# Configuration
# -------------------------------
VIEWPORTS_ROOT = "./viewports"
OUTPUT_PATH = "./features/viewport_features.pt"
# "./features/OIQA/viewport_features.pt" "./features/CVIQ/viewport_features.pt"
IMAGE_NAMES_CSV = "./text/des.csv"  # To get list of image names
# "./text/OIQA/des.csv" "./text/CVIQ/des.csv"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224)
NUM_VIEWPORTS = 20

print(f"Using device: {DEVICE}")

# -------------------------------
# Load image name list
# -------------------------------
df = pd.read_csv(IMAGE_NAMES_CSV)
image_names = df["image_name"].tolist()
print(f"Found {len(image_names)} images.")

# -------------------------------
# Load ViT model and feature extractor
# -------------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()  # Freeze model

# Preprocessing transform (mimics feature_extractor)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# -------------------------------
# Function to load and preprocess image
# -------------------------------
def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0)  # Add batch dim
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# -------------------------------
# Encode all viewports
# -------------------------------
all_viewport_features = []

with torch.no_grad():
    for img_name in image_names:
        base_name = os.path.splitext(img_name)[0]  # Remove .png/.jpg
        viewport_dir = os.path.join(VIEWPORTS_ROOT, base_name)
        
        if not os.path.exists(viewport_dir):
            raise FileNotFoundError(f"Viewport directory not found: {viewport_dir}")

        viewport_features = []
        missing = 0

        for vp_idx in range(NUM_VIEWPORTS):
            vp_path = os.path.join(viewport_dir, f"viewport_{vp_idx:02d}.png")
            if not os.path.exists(vp_path):
                vp_path = os.path.join(viewport_dir, f"viewport_{vp_idx}.png")  # Fallback

            img_tensor = load_image(vp_path)
            if img_tensor is None:
                missing += 1
                # Append zero vector if missing
                feat = torch.zeros(1, 768)
            else:
                img_tensor = img_tensor.to(DEVICE)
                outputs = model(img_tensor)
                feat = outputs.last_hidden_state[:, 0, :]  # CLS token
                feat = feat.cpu()
            viewport_features.append(feat)

        if missing > 0:
            print(f"Warning: {missing} viewport(s) missing for {img_name}")

        # Stack: [20, 1, 768] → [20, 768]
        viewport_tensor = torch.cat(viewport_features, dim=0)  # Shape: [20, 768]
        all_viewport_features.append(viewport_tensor)

# Final shape: [N, 20, 768]
all_viewport_features = torch.stack(all_viewport_features, dim=0)  # [N, 20, 768]
print(f"Final viewport features shape: {all_viewport_features.shape}")

# -------------------------------
# Save features
# -------------------------------
torch.save({
    "image_names": image_names,
    "viewport_features": all_viewport_features,
    "model": MODEL_NAME,
    "num_viewports": NUM_VIEWPORTS,
    "feature_dim": 768
}, OUTPUT_PATH)

print(f"Viewport features saved to {OUTPUT_PATH}")
