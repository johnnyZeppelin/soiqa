# cross_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 50
RESULTS_PATH = "./results/cross_validation_results.txt"

# Feature directories
DATASETS = {
    'OIQA': './features/OIQA/',
    'CVIQ': './features/CVIQ/'
}

# CSV paths (for MOS)
CSV_PATHS = {
    'OIQA': './datasets/OIQA/des.csv',
    'CVIQ': './datasets/CVIQ/des.csv'
}

# Ensure results dir exists
os.makedirs('./results', exist_ok=True)


# -------------------------------
# Load MOS from CSV
# -------------------------------
def load_mos(csv_path):
    df = pd.read_csv(csv_path)
    assert 'MOS' in df.columns, f"MOS column missing in {csv_path}"
    return torch.tensor(df['MOS'].values, dtype=torch.float32)


# -------------------------------
# Load Fusion Features
# -------------------------------
def load_fusion_features(base_path):
    fg3 = torch.load(f"{base_path}/global_fusion_features.pt")['global_fusion_features']
    fl = torch.load(f"{base_path}/local_fusion_features.pt")['local_fusion_features']
    return fg3.float(), fl.float()


# -------------------------------
# AFM Model (same as before)
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


class AFM(nn.Module):
    def __init__(self, global_dim=1344, local_dim=15360, d_model=512):
        super().__init__()
        self.d_model = d_model

        self.global_proj = nn.Sequential(nn.LayerNorm(global_dim), Conv1DBlock(global_dim, d_model))
        self.local_conv1 = nn.Sequential(nn.LayerNorm(local_dim), Conv1DBlock(local_dim, d_model))
        self.local_proj = nn.Sequential(nn.LayerNorm(d_model), Conv1DBlock(d_model, d_model))
        self.self_attn = SelfAttention(dim=d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, FG3, FL):
        B = FG3.shape[0]

        # Project FG3 → K, V
        FG3_norm = torch.nn.functional.layer_norm(FG3, (FG3.shape[-1],))
        K = self.global_proj(FG3_norm)
        V = K

        # Project FL → Q
        FL_conv1 = self.local_conv1(FL)
        FL_norm = torch.nn.functional.layer_norm(FL_conv1, (FL_conv1.shape[-1],))
        Q = self.local_proj(FL_norm)

        # Stack Q, K, V
        fused = torch.stack([Q, K, V], dim=1)
        FA_seq = self.self_attn(fused)
        FA = FA_seq.mean(dim=1)

        pred = self.fc(FA).squeeze(-1)
        return pred, FA


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(model, FG3, FL, mos_values):
    model.eval()
    with torch.no_grad():
        pred, _ = model(FG3, FL)
    pred = pred.cpu().numpy()
    gt = mos_values.cpu().numpy()

    plcc, _ = pearsonr(gt, pred)
    srcc, _ = spearmanr(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred))
    return plcc, srcc, rmse


# -------------------------------
# Training Function
# -------------------------------
def train_on_dataset(train_name, test_name):
    print(f"\n{'='*60}")
    print(f"Training on {train_name} → Testing on {test_name}")
    print(f"{'='*60}")

    # Load training data
    FG3_train, FL_train = load_fusion_features(DATASETS[train_name])
    mos_train = load_mos(CSV_PATHS[train_name])
    FG3_train, FL_train, mos_train = FG3_train.to(DEVICE), FL_train.to(DEVICE), mos_train.to(DEVICE)

    # Load test data
    FG3_test, FL_test = load_fusion_features(DATASETS[test_name])
    mos_test = load_mos(CSV_PATHS[test_name])
    FG3_test, FL_test, mos_test = FG3_test.to(DEVICE), FL_test.to(DEVICE), mos_test.to(DEVICE)

    # Initialize model
    model = AFM(global_dim=1344, local_dim=15360, d_model=512).to(DEVICE)
    optimizer = optim.Adamax(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(FG3_train, FL_train)
        loss = criterion(pred, mos_train)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

    # Final evaluation
    plcc, srcc, rmse = evaluate(model, FG3_test, FL_test, mos_test)
    print(f"Result: PLCC={plcc:.3f}, SRCC={srcc:.3f}, RMSE={rmse:.3f}")

    return plcc, srcc, rmse


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    results = []

    # 1. Train on CVIQ, Test on OIQA
    plcc1, srcc1, rmse1 = train_on_dataset('CVIQ', 'OIQA')
    results.append(f"CVIQ → OIQA | PLCC: {plcc1:.3f} | SRCC: {srcc1:.3f} | RMSE: {rmse1:.3f}")

    # 2. Train on OIQA, Test on CVIQ
    plcc2, srcc2, rmse2 = train_on_dataset('OIQA', 'CVIQ')
    results.append(f"OIQA → CVIQ | PLCC: {plcc2:.3f} | SRCC: {srcc2:.3f} | RMSE: {rmse2:.3f}")

    # Save to file
    with open(RESULTS_PATH, 'w') as f:
        f.write("MMIFN Cross-Dataset Validation Results\n")
        f.write("="*60 + "\n")
        for r in results:
            f.write(r + "\n")
        f.write("\nReproduces Table III from the MMIFN paper.\n")

    print(f"\nResults saved to {RESULTS_PATH}")
