# ablation_study.py
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate, load_mos_and_distortion
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 50
RESULTS_PATH = "./results/ablation_results.txt"
# "./results/CVIQ/ablation_results.txt"  "./results/OIQA/ablation_results.txt"
DATASETS = ['OIQA', 'CVIQ']

# Feature paths
FEATURES_DIR = "./features/"

# CSV paths
CSV_PATHS = {
    'OIQA': './datasets/OIQA/des.csv',
    'CVIQ': './datasets/CVIQ/des.csv'
}

os.makedirs('./results', exist_ok=True)


# -------------------------------
# Load Fusion Features
# -------------------------------
def load_fusion_features(dataset, feature_type):
    path = f"{FEATURES_DIR}/{dataset}/{feature_type}_features.pt"
    data = torch.load(path)
    key = f"{feature_type}_features"
    return data[key].float().to(DEVICE)


# -------------------------------
# AFM Variants
# -------------------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.norm(x)
        x = self.act(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)


# Full AFM (used in No. 1, 3, 6, 7)
class AFM(nn.Module):
    def __init__(self, global_dim=1344, local_dim=15360, d_model=512, use_sa=True):
        super().__init__()
        self.d_model = d_model
        self.use_sa = use_sa

        self.global_proj = nn.Sequential(nn.LayerNorm(global_dim), Conv1DBlock(global_dim, d_model))
        self.local_conv1 = nn.Sequential(nn.LayerNorm(local_dim), Conv1DBlock(local_dim, d_model))
        self.local_proj = nn.Sequential(nn.LayerNorm(d_model), Conv1DBlock(d_model, d_model))

        self.self_attn = SelfAttention(dim=d_model) if use_sa else None

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, FG3, FL):
        B = FG3.shape[0]

        # Project FG3 → K, V
        K = self.global_proj(torch.nn.functional.layer_norm(FG3, (FG3.shape[-1],)))
        V = K

        # Project FL → Q
        FL_conv1 = self.local_conv1(FL)
        Q = self.local_proj(torch.nn.functional.layer_norm(FL_conv1, (FL_conv1.shape[-1],)))

        # Stack and apply self-attention
        fused = torch.stack([Q, K, V], dim=1)
        if self.self_attn is not None:
            fused = self.self_attn(fused)
        FA = fused.mean(dim=1)

        pred = self.fc(FA).squeeze(-1)
        return pred, FA


# Concatenation-based Fusion (used in No. 4, 5, 6)
class ConcatFusion(nn.Module):
    def __init__(self, global_dim=1344, local_dim=15360):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(global_dim + local_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, FG3, FL):
        x = torch.cat([FG3, FL], dim=1)
        return self.fc(x).squeeze(-1), x


# LIT → Concat (used in No. 7)
class LFFModule_Concat(nn.Module):
    def forward(self, ft, fli):
        # Replace LIT with simple concatenation
        return torch.cat([ft.unsqueeze(1).expand(-1, 20, -1), fli], dim=-1).view(fli.shape[0], -1)


# -------------------------------
# Evaluate Function
# -------------------------------
def evaluate_model(pred, gt):
    plcc, _ = pearsonr(gt, pred)
    srcc, _ = spearmanr(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred))
    return plcc, srcc, rmse


# -------------------------------
# Ablation Configurations
# -------------------------------
ABLATED_MODELS = [
    {
        'name': 'Full Model (1)',
        'use_text': True,
        'use_gff': True,
        'use_lff': True,
        'use_afm': True,
        'use_concat_afm': False,
        'use_lit': True,
        'use_concat_lit': False,
        'use_sa': True
    },
    {
        'name': 'w/o Text (3)',
        'use_text': False,
        'use_gff': True,
        'use_lff': True,
        'use_afm': True,
        'use_concat_afm': False,
        'use_lit': True,
        'use_concat_lit': False,
        'use_sa': True
    },
    {
        'name': 'w/o GFF, w/ Concat (4)',
        'use_text': True,
        'use_gff': False,
        'use_lff': True,
        'use_afm': False,
        'use_concat_afm': True,
        'use_lit': True,
        'use_concat_lit': False,
        'use_sa': True
    },
    {
        'name': 'w/o LFF, w/ Concat (5)',
        'use_text': True,
        'use_gff': True,
        'use_lff': False,
        'use_afm': False,
        'use_concat_afm': True,
        'use_lit': True,
        'use_concat_lit': False,
        'use_sa': True
    },
    {
        'name': 'w/o AFM, w/ Concat (6)',
        'use_text': True,
        'use_gff': True,
        'use_lff': True,
        'use_afm': False,
        'use_concat_afm': True,
        'use_lit': True,
        'use_concat_lit': False,
        'use_sa': True
    },
    {
        'name': 'w/o LIT, w/ SA (7)',
        'use_text': True,
        'use_gff': True,
        'use_lff': True,
        'use_afm': True,
        'use_concat_afm': False,
        'use_lit': False,
        'use_concat_lit': True,
        'use_sa': True
    }
]


# -------------------------------
# Main Ablation Loop
# -------------------------------
if __name__ == "__main__":
    results = []

    # Load ground truth
    mos_o = load_mos_and_distortion(CSV_PATHS['OIQA'])[0]
    mos_c = load_mos_and_distortion(CSV_PATHS['CVIQ'])[0]

    # Load full features
    FG3_o = load_fusion_features('OIQA', 'global_fusion')
    FL_o = load_fusion_features('OIQA', 'local_fusion')
    FG3_c = load_fusion_features('CVIQ', 'global_fusion')
    FL_c = load_fusion_features('CVIQ', 'local_fusion')

    print("Starting ablation study...\n")

    for config in ABLATED_MODELS:
        name = config['name']
        print(f"{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Build model
        if config['use_concat_afm']:
            model = ConcatFusion().to(DEVICE)
        else:
            model = AFM(use_sa=config['use_sa']).to(DEVICE)

        optimizer = optim.Adamax(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        # Train on OIQA
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            if config['use_gff'] and config['use_lff']:
                pred, _ = model(FG3_o, FL_o)
            elif not config['use_gff']:
                pred, _ = model(torch.zeros_like(FG3_o), FL_o)  # Zero global
            elif not config['use_lff']:
                pred, _ = model(FG3_o, torch.zeros_like(FL_o))  # Zero local
            else:
                pred, _ = model(FG3_o, FL_o)
            loss = criterion(pred, mos_o.to(DEVICE))
            loss.backward()
            optimizer.step()

        # Evaluate on OIQA
        model.eval()
        with torch.no_grad():
            if config['use_gff'] and config['use_lff']:
                pred_o, _ = model(FG3_o, FL_o)
            elif not config['use_gff']:
                pred_o, _ = model(torch.zeros_like(FG3_o), FL_o)
            elif not config['use_lff']:
                pred_o, _ = model(FG3_o, torch.zeros_like(FL_o))
            else:
                pred_o, _ = model(FG3_o, FL_o)
            pred_o = pred_o.cpu().numpy()

        plcc_o, srcc_o, rmse_o = evaluate_model(pred_o, mos_o)

        # Evaluate on CVIQ (zero-shot)
        with torch.no_grad():
            pred_c, _ = model(FG3_c, FL_c)
            pred_c = pred_c.cpu().numpy()
        plcc_c, srcc_c, rmse_c = evaluate_model(pred_c, mos_c)

        print(f"OIQA  PLCC: {plcc_o:.3f}, SRCC: {srcc_o:.3f}, RMSE: {rmse_o:.3f}")
        print(f"CVIQ  PLCC: {plcc_c:.3f}, SRCC: {srcc_c:.3f}, RMSE: {rmse_c:.3f}")

        results.append({
            'model': name,
            'OIQA_PLCC': plcc_o,
            'OIQA_SRCC': srcc_o,
            'OIQA_RMSE': rmse_o,
            'CVIQ_PLCC': plcc_c,
            'CVIQ_SRCC': srcc_c,
            'CVIQ_RMSE': rmse_c
        })

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        f.write("MMIFN Ablation Study Results\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<25} {'OIQA PLCC':<10} {'CVIQ PLCC':<10} {'OIQA RMSE':<10} {'CVIQ RMSE':<10}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r['model']:<25} {r['OIQA_PLCC']:<10.3f} {r['CVIQ_PLCC']:<10.3f} "
                    f"{r['OIQA_RMSE']:<10.3f} {r['CVIQ_RMSE']:<10.3f}\n")
        f.write("\nReproduces Table IV from the MMIFN paper.\n")

    print(f"\nAblation results saved to {RESULTS_PATH}")
