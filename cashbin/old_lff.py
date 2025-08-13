# lff_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Configuration
# -------------------------------
TEXT_DIM = 768
VIEWPORT_DIM = 768
NUM_VIEWPORTS = 20
OUTPUT_PATH = "./features/local_fusion_features.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# -------------------------------
# Local Image-Text (LIT) Fusion Module
# -------------------------------
class LITFusion(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.mlp1 = nn.Linear(feature_dim, feature_dim)
        self.mlp2 = nn.Linear(feature_dim, feature_dim)
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),  # Simulate Conv Block
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, ft, fli):
        # ft: [B, 768] (text feature)
        # fli: [B, 768] (single viewport feature)

        # Normalize viewport features
        fli_norm = F.softmax(fli, dim=-1)  # [B, 768]

        # Eq. (13): ˜fti = ft ⊗ MLP(SoftMax(fli))
        term1 = ft * self.mlp1(fli_norm)  # [B, 768]

        # Eq. (14): ˜fti = ˜fti ⊕ MLP(SoftMax(fli))
        term2 = self.mlp2(fli_norm)       # [B, 768]
        fused = term1 + term2             # [B, 768]

        # Eq. (15): fvti = Conv(ReLU(˜fti))
        fused = self.relu(fused)
        fused = fused.unsqueeze(1)        # [B, 1, 768] for Conv1d
        fused = self.conv_block(fused)    # [B, 1, 768]
        fused = fused.squeeze(1)          # [B, 768]

        return fused

# -------------------------------
# LFF Module (applies LIT to all 20 viewports)
# -------------------------------
class LFFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lit = LITFusion()

    def forward(self, text_features, viewport_features):
        # text_features: [N, 768]
        # viewport_features: [N, 20, 768]
        B = text_features.shape[0]
        fused_viewports = []

        for i in range(NUM_VIEWPORTS):
            fli = viewport_features[:, i, :]  # [N, 768]
            fvti = self.lit(text_features, fli)  # [N, 768]
            fused_viewports.append(fvti)

        # Concatenate all 20 fused viewport features
        FL = torch.cat(fused_viewports, dim=1)  # [N, 20 * 768] = [N, 15360]
        return FL

# -------------------------------
# Load Features
# -------------------------------
def load_feature(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data

text_features = load_feature("./features/text_features.pt", "text_features").to(DEVICE)
viewport_features = load_feature("./features/viewport_features.pt", "viewport_features").to(DEVICE)

print(f"Loaded text features: {text_features.shape}")
print(f"Loaded viewport features: {viewport_features.shape}")

# -------------------------------
# Apply LFF Module
# -------------------------------
model = LFFModule().to(DEVICE)
model.eval()

with torch.no_grad():
    local_fusion_features = model(text_features, viewport_features)

print(f"Local fusion features shape: {local_fusion_features.shape}")  # Should be [N, 15360]

# -------------------------------
# Save Output
# -------------------------------
torch.save({
    "image_names": torch.load("./features/text_features.pt")["image_names"],
    "local_fusion_features": local_fusion_features.cpu(),
    "feature_dim": local_fusion_features.shape[1],
    "description": "Output of LFF module: concatenated fusion of text + 20 viewports via LIT"
}, OUTPUT_PATH)

print(f"Local fusion features saved to {OUTPUT_PATH}")
