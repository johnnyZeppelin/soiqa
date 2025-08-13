# afm_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Configuration
# -------------------------------
GLOBAL_DIM = 1344
LOCAL_DIM = 15360
D_MODEL = 512  # Latent dimension for attention
OUTPUT_PATH = "./checkpoints/afm_model.pth"
PREDICTIONS_PATH = "./results/predictions.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# -------------------------------
# Aggregated Fusion Module (AFM)
# -------------------------------
class AFM(nn.Module):
    def __init__(self, global_dim=1344, local_dim=15360, d_model=512, dropout=0.1):
        super().__init__()
        # Projection layers
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.local_proj = nn.Sequential(
            nn.Linear(local_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Self-Attention expects (B, L, D)
        # We treat global and local as sequences of length 1
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=dropout, batch_first=True
        )

        # Final prediction head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, FG3, FL):
        # FG3: [B, 1344], FL: [B, 15360]
        B = FG3.shape[0]

        # Project to shared space
        FG3_proj = self.global_proj(FG3)  # [B, 512]
        FL_proj = self.local_proj(FL)     # [B, 512]

        # Add sequence dim
        FG3_seq = FG3_proj.unsqueeze(1)  # [B, 1, D]
        FL_seq = FL_proj.unsqueeze(1)    # [B, 1, D]

        # AFM: K, V ← FG3, Q ← FL
        # Self-attention: Q, K, V all from same space
        # But here: we use FL as Q, FG3 as K, V → like cross-attention
        attn_out, _ = self.self_attention(
            query=FL_seq,   # [B, 1, D]
            key=FG3_seq,    # [B, 1, D]
            value=FG3_seq   # [B, 1, D]
        )  # [B, 1, D]

        FA = attn_out.squeeze(1)  # [B, D]

        # Final prediction
        pred = self.fc(FA).squeeze(-1)  # [B]

        return pred, FA

# -------------------------------
# Load Features
# -------------------------------
def load_feature(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data

# Load fusion features
global_fusion = load_feature("./features/global_fusion_features.pt", "global_fusion_features").to(DEVICE)
local_fusion = load_feature("./features/local_fusion_features.pt", "local_fusion_features").to(DEVICE)

print(f"Loaded global fusion: {global_fusion.shape}")  # [N, 1344]
print(f"Loaded local fusion: {local_fusion.shape}")    # [N, 15360]

# -------------------------------
# Simulate Training (if MOS available)
# -------------------------------
# Load MOS from CSV
import pandas as pd
df = pd.read_csv("./text/des.csv")
mos_values = torch.tensor(df["MOS"].values, dtype=torch.float32).to(DEVICE)  # Assuming 'MOS' column exists

# Initialize model
model = AFM(global_dim=GLOBAL_DIM, local_dim=LOCAL_DIM, d_model=D_MODEL).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop (single epoch for demo)
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    pred, _ = model(global_fusion, local_fusion)
    loss = criterion(pred, mos_values)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training completed.")

# Save model
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"AFM model saved to {OUTPUT_PATH}")

# Inference
model.eval()
with torch.no_grad():
    predictions, attention_features = model(global_fusion, local_fusion)

# Save predictions
torch.save({
    "predictions": predictions.cpu(),
    "ground_truth": mos_values.cpu(),
    "attention_features": attention_features.cpu(),
    "model_config": {
        "d_model": D_MODEL,
        "global_dim": GLOBAL_DIM,
        "local_dim": LOCAL_DIM
    }
}, PREDICTIONS_PATH)

print(f"Predictions saved to {PREDICTIONS_PATH}")
