# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from afm_module import AFM
from gff_module import GFFModule
from lff_module import LFFModule

# -------------------------------
# Configuration
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 100
SAVE_PATH = "./checkpoints/mmifn_best.pth"
# "./checkpoints/OIQA/mmifn_best.pth" "./checkpoints/CVIQ/mmifn_best.pth"
FEATURES_DIR = "./features/"

# -------------------------------
# Load Pre-extracted Features
# -------------------------------
def load_feature(path, key=None):
    data = torch.load(path)
    if isinstance(data, dict):
        return data[key] if key in data else data[list(data.keys())[0]]
    return data

# Load fusion features
FG3 = load_feature(f"{FEATURES_DIR}global_fusion_features.pt", "global_fusion_features").to(DEVICE)
FL = load_feature(f"{FEATURES_DIR}local_fusion_features.pt", "local_fusion_features").to(DEVICE)

# Load MOS
df = pd.read_csv("./text/des.csv")
if 'MOS' not in df.columns:
    raise ValueError("Please ensure 'MOS' column exists in des.csv")
mos_values = torch.tensor(df["MOS"].values, dtype=torch.float32).to(DEVICE)

# -------------------------------
# Initialize Model
# -------------------------------
model = AFM(global_dim=1344, local_dim=15360, d_model=512).to(DEVICE)
optimizer = optim.Adamax(model.parameters(), lr=LR)  # AdaMax as in paper
criterion = nn.MSELoss()

# -------------------------------
# Training Loop
# -------------------------------
best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred, _ = model(FG3, FL)
    loss = criterion(pred, mos_values)
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), SAVE_PATH)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"Training completed. Best model saved to {SAVE_PATH}")
