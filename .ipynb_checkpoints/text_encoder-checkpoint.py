# text_encoder.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
CSV_PATH = "./text/des.csv"
OUTPUT_FEATURES_PATH = "./features/text_features.pt"  # Save as PyTorch tensor
# "./features/OIQA/text_features.pt" "./features/CVIQ/text_features.pt"
MODEL_NAME = "THUDM/ImageReward-v1.0"
SUBFOLDER = "text_encoder"
MAX_LENGTH = 128  # Max tokens for BERT
BATCH_SIZE = 16   # For faster inference

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} text descriptions.")

# -------------------------------
# Load tokenizer and model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder=SUBFOLDER)
text_encoder = AutoModel.from_pretrained(MODEL_NAME, subfolder=SUBFOLDER).to(device)
text_encoder.eval()  # Freeze the model

# -------------------------------
# Encode function
# -------------------------------
def encode_texts(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**inputs)
        # Use [CLS] token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

    return cls_embeddings.cpu()

# -------------------------------
# Batched encoding
# -------------------------------
all_embeddings = []
image_names = []

for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    texts = batch["description"].tolist()
    names = batch["image_name"].tolist()

    embeddings = encode_texts(texts)
    all_embeddings.append(embeddings)
    image_names.extend(names)

    print(f"Encoded batch {i//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1}")

# Concatenate all
all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Final text features shape: {all_embeddings.shape}")  # Should be [N, 768]

# -------------------------------
# Save outputs
# -------------------------------
# Option 1: Save as PyTorch dict with image names
torch.save({
    "image_names": image_names,
    "text_features": all_embeddings,
    "model": MODEL_NAME,
    "subfolder": SUBFOLDER
}, OUTPUT_FEATURES_PATH)

print(f"Text features saved to {OUTPUT_FEATURES_PATH}")
