# gff_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
TEXT_FEATURE_DIM = 768
VISUAL_FEATURE_DIMS = {
    'f2': 192,
    'f3': 384,
    'f4': 768
}
OUTPUT_PATH = "./features/global_fusion_features.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# -------------------------------
# 1D Conv for Multi-scale Text Features
# -------------------------------
class TextConv1D(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.unsqueeze(1)  # [B, 1, D] since L=1 (global text feature)
        x = self.conv(x)    # [B, D_out, 1]
        x = x.squeeze(-1)   # [B, D_out]
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------------
# Bi-LSTM Block
# -------------------------------
class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)  # fuse bidir
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [B, SeqLen, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]

        out, _ = self.lstm(x)  # [B, SeqLen, 2*H]
        out = self.proj(out)   # [B, SeqLen, D]
        out = self.norm(out + x)  # residual
        return out.squeeze(1)  # back to [B, D] if SeqLen=1


# -------------------------------
# Multihead Cross-Attention Fusion (MCA)
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # Both: [B, D] -> [B, 1, D]
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)

        # Self-Attention on key_value (Eq. 10-11)
        sa = nn.MultiheadAttention(embed_dim=key_value.shape[-1], num_heads=8, batch_first=True)
        key_value_attn, _ = sa(key_value, key_value, key_value)
        key_value = key_value_attn

        # Cross-Attention: query attends to key_value
        attn_out, _ = self.multihead_attn(query, key_value, key_value)  # [B, 1, D]
        out = attn_out.squeeze(1)  # [B, D]
        return self.norm(out)


# -------------------------------
# MGMF Block (Multi-scale Global Multimodal Fusion)
# -------------------------------
class MGMFBlock(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.bilstm_text = BiLSTMBlock(visual_dim)
        self.bilstm_visual = BiLSTMBlock(visual_dim)
        self.attn_fusion = AttentionFusion(visual_dim)

    def forward(self, ft, fv):
        # Project both to same dim
        ft = self.text_proj(ft)  # [B, D_v]
        fv = self.visual_proj(fv)  # [B, H, W, C] -> need to flatten spatial dims

        B, C, H, W = fv.shape
        fv = fv.view(B, H * W, C)  # [B, L, C]

        # Apply Bi-LSTM
        ft_enhanced = self.bilstm_text(ft)
        fv_enhanced = self.bilstm_visual(fv.mean(dim=1))  # global avg over spatial

        # Attention Fusion
        fused = self.attn_fusion(ft_enhanced, fv_enhanced)
        return fused


# -------------------------------
# GFF Module
# -------------------------------
class GFFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mgmf1 = MGMFBlock(TEXT_FEATURE_DIM, VISUAL_FEATURE_DIMS['f2'])
        self.mgmf2 = MGMFBlock(TEXT_FEATURE_DIM, VISUAL_FEATURE_DIMS['f3'])
        self.mgmf3 = MGMFBlock(TEXT_FEATURE_DIM, VISUAL_FEATURE_DIMS['f4'])
        self.conv1d = TextConv1D()  # to get multi-scale text

    def forward(self, ft, f2, f3, f4):
        # ft: [B, 768]
        # f2: [B, 192, H2, W2], f3: [B, 384, H3, W3], f4: [B, 768, H4, W4]

        # Enhance text feature
        ft_conv = self.conv1d(ft)  # [B, 768] → [B, 768]

        # Level 1: f2
        fg1 = self.mgmf1(ft_conv, f2)  # [B, 192]

        # Level 2: f3 + concat with fg1
        fg2_local = self.mgmf2(ft_conv, f3)  # [B, 384]
        fg2 = torch.cat([fg1, fg2_local], dim=1)  # [B, 192+384=576]

        # Level 3: f4 + concat with fg2
        fg3_local = self.mgmf3(ft_conv, f4)  # [B, 768]
        fg3 = torch.cat([fg2, fg3_local], dim=1)  # [B, 576+768=1344]

        return fg3  # Final global fusion feature


# -------------------------------
# Load Features and Run GFF
# -------------------------------
def load_tensor(path, key):
    data = torch.load(path)
    return data[key] if isinstance(data, dict) else data


# Load all features
text_features = load_tensor("./features/text_features.pt", "text_features").to(DEVICE)  # [N, 768]
f2 = load_tensor("./features/global_features.pt", "f2").to(DEVICE)  # [N, 192, H, W]
f3 = load_tensor("./features/global_features.pt", "f3").to(DEVICE)  # [N, 384, H, W]
f4 = load_tensor("./features/global_features.pt", "f4").to(DEVICE)  # [N, 768, H, W]

print(f"Loaded text: {text_features.shape}")
print(f"Loaded f2: {f2.shape}, f3: {f3.shape}, f4: {f4.shape}")

# Initialize and run GFF
model = GFFModule().to(DEVICE)
model.eval()

with torch.no_grad():
    global_fusion_features = model(text_features, f2, f3, f4)

print(f"Global fusion features shape: {global_fusion_features.shape}")  # Expected: [N, 1344]

# Save result
torch.save({
    "image_names": torch.load("./features/text_features.pt")["image_names"],
    "global_fusion_features": global_fusion_features.cpu(),
    "feature_dim": global_fusion_features.shape[1],
    "description": "Output of GFF module: multi-scale fused text + global visual features"
}, OUTPUT_PATH)

print(f"Global fusion features saved to {OUTPUT_PATH}")
