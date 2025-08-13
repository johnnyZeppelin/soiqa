# gff_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Configuration
# -------------------------------
TEXT_DIM = 768
VISUAL_DIMS = {
    'f2': 192,  # Stage 2
    'f3': 384,  # Stage 3
    'f4': 768,  # Stage 4
}
OUTPUT_PATH = "./features/global_fusion_features.pt"  # "./features/OIQA/global_fusion_features.pt" "./features/CVIQ/global_fusion_features.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 768  # Shared latent dim for fusion

print(f"Using device: {DEVICE}")


# -------------------------------
# Multi-Scale Text Feature Extraction
# -------------------------------
class MultiScaleTextEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # Three different "conv" projections (simulated as linear layers)
        self.proj1 = nn.Linear(input_dim, VISUAL_DIMS['f2'])
        self.proj2 = nn.Linear(input_dim, VISUAL_DIMS['f3'])
        self.proj3 = nn.Linear(input_dim, VISUAL_DIMS['f4'])
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.GELU()

    def forward(self, ft):
        # ft: [B, 768]
        ft = self.norm(ft)
        ft = self.act(ft)
        # Return multi-scale projections
        return {
            'f2': self.proj1(ft),  # [B, 192]
            'f3': self.proj2(ft),  # [B, 384]
            'f4': self.proj3(ft),  # [B, 768]
        }


# -------------------------------
# Bi-LSTM Block (for sequence modeling)
# -------------------------------
class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [B, L, D]
        out, _ = self.lstm(x)  # [B, L, 2*H]
        out = self.proj(out)   # [B, L, D]
        out = self.norm(out + x)
        return out  # [B, L, D]


# -------------------------------
# Self-Attention Module (for visual features)
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
# Multihead Cross-Attention Fusion (MCA)
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # query: [B, 1, D], key_value: [B, L, D]
        out, _ = self.attn(query, key_value, key_value)  # Cross-attention
        return self.norm(out + query)  # [B, 1, D]


# -------------------------------
# MGMF Block (Multi-scale Global Multimodal Fusion)
# -------------------------------
class MGMFBlock(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super().__init__()
        self.visual_dim = visual_dim

        # Bi-LSTM for visual features (sequence of spatial tokens)
        self.bilstm_visual = BiLSTMBlock(visual_dim)
        self.sa_visual = SelfAttention(visual_dim)

        # Bi-LSTM for text (treated as sequence of one token)
        self.bilstm_text = BiLSTMBlock(text_dim)

        # Attention fusion
        self.attn_fusion = AttentionFusion(visual_dim)

    def forward(self, ft, fv):
        # ft: [B, D_t]  (text feature)
        # fv: [B, C, H, W] → [B, L, C]

        B, C, H, W = fv.shape
        fv = fv.view(B, H * W, C)  # [B, L, C]

        # Step 1: Enhance visual features
        fv = self.bilstm_visual(fv)       # [B, L, C]
        fv = self.sa_visual(fv)           # [B, L, C]

        # Step 2: Enhance text features
        ft_seq = ft.unsqueeze(1)          # [B, 1, D_t]
        ft_seq = self.bilstm_text(ft_seq) # [B, 1, D_t]

        # Project text to visual dim
        ft_proj = nn.Linear(ft_seq.shape[-1], self.visual_dim).to(ft_seq.device)
        ft_seq = ft_proj(ft_seq)          # [B, 1, C]

        # Step 3: Cross-Attention Fusion
        fused = self.attn_fusion(ft_seq, fv)  # [B, 1, C]
        return fused.squeeze(1)  # [B, C]


# -------------------------------
# GFF Module
# -------------------------------
class GFFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = MultiScaleTextEncoder()
        self.mgmf1 = MGMFBlock(text_dim=768, visual_dim=192)
        self.mgmf2 = MGMFBlock(text_dim=768, visual_dim=384)
        self.mgmf3 = MGMFBlock(text_dim=768, visual_dim=768)

    def forward(self, ft, f2, f3, f4):
        # ft: [B, 768]
        # f2: [B, 192, H2, W2], f3: [B, 384, H3, W3], f4: [B, 768, H4, W4]

        # Get multi-scale text features
        ft_scales = self.text_encoder(ft)

        # Level 1: f2
        fg1 = self.mgmf1(ft_scales['f2'], f2)  # [B, 192]

        # Level 2: f3 + concat with fg1
        fg2_local = self.mgmf2(ft_scales['f3'], f3)  # [B, 384]
        fg2 = torch.cat([fg1, fg2_local], dim=1)     # [B, 576]

        # Level 3: f4 + concat with fg2
        fg3_local = self.mgmf3(ft_scales['f4'], f4)  # [B, 768]
        fg3 = torch.cat([fg2, fg3_local], dim=1)     # [B, 1344]

        return fg3


# -------------------------------
# Load Features and Run GFF
# -------------------------------
def load_feature(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data


# Load text and global visual features
text_features = load_feature("./features/text_features.pt", "text_features").to(DEVICE)  # [N, 768]
# "./features/OIQA/text_features.pt" "./features/CVIQ/text_features.pt"
f2 = load_feature("./features/global_features.pt", "f2").to(DEVICE)  # [N, 192, H, W]
# "./features/OIQA/global_features.pt" "./features/CVIQ/global_features.pt"
f3 = load_feature("./features/global_features.pt", "f3").to(DEVICE)  # [N, 384, H, W]
# "./features/OIQA/global_features.pt" "./features/CVIQ/global_features.pt"
f4 = load_feature("./features/global_features.pt", "f4").to(DEVICE)  # [N, 768, H, W]
# "./features/OIQA/global_features.pt" "./features/CVIQ/global_features.pt"

print(f"Loaded text: {text_features.shape}")
print(f"Loaded f2: {f2.shape}, f3: {f3.shape}, f4: {f4.shape}")

# Initialize and run GFF
model = GFFModule().to(DEVICE)
model.eval()

with torch.no_grad():
    global_fusion_features = model(text_features, f2, f3, f4)

print(f"Global fusion features shape: {global_fusion_features.shape}")  # Should be [N, 1344]

# Save result
torch.save({
    "image_names": torch.load("./features/text_features.pt")["image_names"],
    # "./features/OIQA/text_features.pt" "./features/CVIQ/text_features.pt"
    "global_fusion_features": global_fusion_features.cpu(),
    "feature_dim": global_fusion_features.shape[1],
    "description": "Output of GFF module: multi-scale fused text + global visual features"
}, OUTPUT_PATH)

print(f"Global fusion features saved to {OUTPUT_PATH}")
