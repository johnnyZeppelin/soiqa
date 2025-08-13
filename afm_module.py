# afm_module.py
import torch
import torch.nn as nn

# -------------------------------
# Configuration
# -------------------------------
GLOBAL_FUSION_DIM = 1344   # From GFF: [N, 1344]
LOCAL_FUSION_DIM = 15360   # From LFF: [N, 15360]
D_MODEL = 512              # Latent dimension for attention


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
        x = x.unsqueeze(-1)           # [B, D] -> [B, D, 1]
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
    def __init__(self, global_dim=GLOBAL_FUSION_DIM, local_dim=LOCAL_FUSION_DIM, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model

        # Eq. (17): K = V = Conv(LN(FG3))
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            Conv1DBlock(global_dim, d_model)
        )

        # Eq. (18): Q = Conv(LN(Conv(FL)))
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

        # Project FG3 → K, V
        FG3_norm = torch.nn.functional.layer_norm(FG3, (FG3.shape[-1],))
        K = self.global_proj(FG3_norm)  # [B, D_model]
        V = K  # [B, D_model]

        # Project FL → Q
        FL_conv1 = self.local_conv1(FL)                    # First Conv
        FL_norm = torch.nn.functional.layer_norm(FL_conv1, (FL_conv1.shape[-1],))
        Q = self.local_proj(FL_norm)                       # Second Conv

        # Stack Q, K, V into a sequence for self-attention
        fused = torch.stack([Q, K, V], dim=1)  # [B, 3, D_model]

        # Eq. (19): FA = SelfAttn(K, Q, V) → applied via self-attention on fused sequence
        FA_seq = self.self_attn(fused)         # [B, 3, D_model]
        FA = FA_seq.mean(dim=1)                # [B, D_model]

        # Final prediction
        pred = self.fc(FA).squeeze(-1)         # [B]

        return pred, FA
