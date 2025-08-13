# afm_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Configuration
# -------------------------------
GLOBAL_FUSION_DIM = 1344   # From GFF: [N, 1344]
LOCAL_FUSION_DIM = 15360   # From LFF: [N, 15360]
D_MODEL = 512              # Latent dimension for attention
OUTPUT_PATH = "./checkpoints/afm_model.pth"
PREDICTIONS_PATH = "./results/predictions.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# -------------------------------
# Conv1D Block (simulate paper's "Conv")
# -------------------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, D] -> [B, D, 1]
        x = x.unsqueeze(-1)           # [B, D, 1]
        x = self.conv(x)              # [B, D_out, 1]
        x = x.squeeze(-1)             # [B, D_out]
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------------
# Self-Attention Module
# -------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, L, D]
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        return self.norm(attn_out + x)


# -------------------------------
# Aggregated Fusion Module (AFM)
# -------------------------------
class AFM(nn.Module):
    def __init__(self, global_dim=1344, local_dim=15360, d_model=512):
        super().__init__()
        self.d_model = d_model

        # Projection for FG3: Conv(LN(FG3))
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            Conv1DBlock(global_dim, d_model)  # Conv
        )

        # Projection for FL: Conv(LN(Conv(FL)))
        self.local_conv1 = nn.Sequential(
            nn.LayerNorm(local_dim),
            Conv1DBlock(local_dim, d_model)
        )
        self.local_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            Conv1DBlock(d_model, d_model)
        )

        # Final self-attention
        self.self_attn = SelfAttention(dim=d_model)

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, FG3, FL):
        """
        FG3: [B, 1344] — global fusion feature
        FL:  [B, 15360] — local fusion feature
        """
        B = FG3.shape[0]

        # Eq. (17): K = V = Conv(LN(FG3))
        FG3_norm = torch.nn.functional.layer_norm(FG3, (FG3.shape[-1],))
        K = self.global_proj(FG3_norm)  # [B, D_model]
        V = K  # [B, D_model]

        # Eq. (18): Q = Conv(LN(Conv(FL)))
        FL_conv1 = self.local_conv1(FL)                    # First Conv
        FL_norm = torch.nn.functional.layer_norm(FL_conv1, (FL_conv1.shape[-1],))
        Q = self.local_proj(FL_norm)                       # Second Conv

        # Reshape for self-attention: treat Q, K, V as sequence of 3 tokens
        # [B, D] -> [B, 3, D]
        fused = torch.stack([Q, K, V], dim=1)  # [B, 3, D_model]

        # Eq. (19): FA = SelfAttn(K, Q, V)
        # But since it's self-attention, we apply it to the fused sequence
        FA_seq = self.self_attn(fused)         # [B, 3, D_model]
        FA = FA_seq.mean(dim=1)                # [B, D_model], or use Q token

        # Final prediction
        pred = self.fc(FA).squeeze(-1)         # [B]

        return pred, FA


# -------------------------------
# Load Features and Run AFM
# -------------------------------
def load_feature(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data


# Load fusion features
global_fusion = load_feature("./features/global_fusion_features.pt", "global_fusion_features").to(DEVICE)  # [N, 1344]
local_fusion = load_feature("./features/local_fusion_features.pt", "local_fusion_features").to(DEVICE)      # [N, 15360]

print(f"Loaded global fusion: {global_fusion.shape}")
print(f"Loaded local fusion: {local_fusion.shape}")

# Check for MOS
import pandas as pd
df = pd.read_csv("./text/des.csv")
if 'MOS' not in df.columns:
    raise ValueError("Please ensure 'MOS' column exists in des.csv")

mos_values = torch.tensor(df["MOS"].values, dtype=torch.float32).to(DEVICE)


# Initialize model
model = AFM(global_dim=GLOBAL_FUSION_DIM, local_dim=LOCAL_FUSION_DIM, d_model=D_MODEL).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
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
        "global_dim": GLOBAL_FUSION_DIM,
        "local_dim": LOCAL_FUSION_DIM
    }
}, PREDICTIONS_PATH)

print(f"Predictions saved to {PREDICTIONS_PATH}")
