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
# "./features/OIQA/local_fusion_features.pt" "./features/CVIQ/local_fusion_features.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# -------------------------------
# MLP Block (as in ViT)
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# -------------------------------
# Conv Block (simulate paper's "Conv Block")
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, D] -> [B, 1, D]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(1)  # [B, D]
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------------
# Local Image-Text (LIT) Fusion Module
# -------------------------------
class LITFusion(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim

        # Two MLPs for visual feature transformation
        self.mlp1 = MLP(feature_dim, 768, feature_dim)
        self.mlp2 = MLP(feature_dim, 768, feature_dim)

        # Final Conv Block
        self.conv_block = ConvBlock(feature_dim)

        # Optional layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, ft, fli):
        """
        ft: [B, 768] — text feature
        fli: [B, 768] — i-th viewport feature
        """
        # Normalize viewport features along feature dim
        fli_softmax = F.softmax(fli, dim=-1)  # [B, 768]

        # Eq. (13): ˜fti = ft ⊗ MLP(SoftMax(fli))
        term1 = self.mlp1(fli_softmax)        # [B, 768]
        fused = ft * term1                    # [B, 768]

        # Eq. (14): ˜fti = ˜fti ⊕ MLP(SoftMax(fli))
        term2 = self.mlp2(fli_softmax)        # [B, 768]
        fused = fused + term2                 # [B, 768]

        # Eq. (15): fvti = Conv(ReLU(˜fti))
        fused = F.relu(fused)
        fused = self.conv_block(fused)        # [B, 768]

        return fused  # [B, 768]


# -------------------------------
# LFF Module (applies LIT to all 20 viewports)
# -------------------------------
class LFFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lit = LITFusion(feature_dim=768)

    def forward(self, text_features, viewport_features):
        """
        text_features: [B, 768]
        viewport_features: [B, 20, 768]
        """
        B, N, D = viewport_features.shape
        assert N == 20, f"Expected 20 viewports, got {N}"

        fused_viewports = []

        for i in range(N):
            fli = viewport_features[:, i, :]  # [B, 768]
            fvti = self.lit(text_features, fli)  # [B, 768]
            fused_viewports.append(fvti)

        # Concatenate all 20 fused features
        FL = torch.cat(fused_viewports, dim=1)  # [B, 20 * 768] = [B, 15360]
        return FL


# -------------------------------
# Load Features and Run LFF
# -------------------------------
def load_feature(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data


# Load text and viewport features
text_features = load_feature("./features/text_features.pt", "text_features").to(DEVICE)  # [N, 768]
# "./features/OIQA/text_features.pt" "./features/CVIQ/text_features.pt"
viewport_features = load_feature("./features/viewport_features.pt", "viewport_features").to(DEVICE)  # [N, 20, 768]
# "./features/OIQA/viewport_features.pt" "./features/CVIQ/viewport_features.pt"

print(f"Loaded text features: {text_features.shape}")
print(f"Loaded viewport features: {viewport_features.shape}")

# Initialize and run LFF
model = LFFModule().to(DEVICE)
model.eval()

with torch.no_grad():
    local_fusion_features = model(text_features, viewport_features)

print(f"Local fusion features shape: {local_fusion_features.shape}")  # Expected: [N, 15360]

# Save result
torch.save({
    "image_names": torch.load("./features/text_features.pt")["image_names"],
    # "./features/OIQA/text_features.pt" "./features/CVIQ/text_features.pt"
    "local_fusion_features": local_fusion_features.cpu(),
    "feature_dim": local_fusion_features.shape[1],
    "description": "Output of LFF module: fused text + 20 viewports via LIT"
}, OUTPUT_PATH)

print(f"Local fusion features saved to {OUTPUT_PATH}")
