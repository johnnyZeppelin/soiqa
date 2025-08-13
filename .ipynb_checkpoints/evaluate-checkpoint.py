# evaluate.py
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import argparse
import os

# -------------------------------
# Configuration
# -------------------------------
PREDICTIONS_PATH = "./results/predictions.pt"
# "./results/OIQA/predictions.pt" "./results/CVIQ/predictions.pt"
TEST_CSV_PATH = "./text/des.csv"  # Must contain 'MOS' and 'distortion_type' columns
# "./text/OIQA/des.csv" "./text/CVIQ/des.csv"
OUTPUT_LOG = "./results/evaluation_results.txt"
# "./results/OIQA/evaluation_results.txt" "./results/CVIQ/evaluation_results.txt"

# Supported distortion types (from OIQA & CVIQ)
DISTORTION_TYPES = {
    'OIQA': ['JPEG', 'JP2K', 'GN', 'GB'],
    'CVIQ': ['JPEG', 'H.264/AVC', 'H.265/HEVC']
}

# Dataset type: auto-detect or set manually
DATASET = None  # 'OIQA' or 'CVIQ' — will be inferred if not set


def load_predictions(path):
    """Load predicted scores and ground truth."""
    data = torch.load(path)
    pred = data['predictions'].cpu().numpy()
    gt = data['ground_truth'].cpu().numpy()
    return pred, gt


def load_mos_and_distortion(csv_path):
    """Load MOS and distortion type for each image."""
    df = pd.read_csv(csv_path)
    if 'MOS' not in df.columns:
        raise ValueError("CSV must contain 'MOS' column.")
    if 'distortion_type' not in df.columns:
        raise ValueError("CSV must contain 'distortion_type' column.")
    return df['MOS'].values, df['distortion_type'].values, df['image_name'].values


def compute_metrics(gt, pred):
    """Compute PLCC, SRCC, RMSE."""
    plcc, _ = pearsonr(gt, pred)
    srcc, _ = spearmanr(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred))
    return plcc, srcc, rmse


def evaluate_per_distortion(pred, gt, distortion_types, dataset_name):
    """Evaluate metrics per distortion type."""
    results = {}
    types = DISTORTION_TYPES[dataset_name]

    for d_type in types:
        mask = [d_type.lower() in d.lower() for d in distortion_types]
        if not any(mask):
            print(f"Warning: No samples found for distortion type: {d_type}")
            results[d_type] = {'PLCC': 0.0, 'SRCC': 0.0, 'RMSE': float('inf')}
            continue

        gt_d = gt[mask]
        pred_d = pred[mask]
        plcc, srcc, rmse = compute_metrics(gt_d, pred_d)
        results[d_type] = {'PLCC': plcc, 'SRCC': srcc, 'RMSE': rmse}
        print(f"{d_type:12} | PLCC: {plcc:.3f} | SRCC: {srcc:.3f} | RMSE: {rmse:.3f}")
    return results


def evaluate_overall(pred, gt):
    """Evaluate overall metrics."""
    plcc, srcc, rmse = compute_metrics(gt, pred)
    print(f"{'Overall':12} | PLCC: {plcc:.3f} | SRCC: {srcc:.3f} | RMSE: {rmse:.3f}")
    return {'PLCC': plcc, 'SRCC': srcc, 'RMSE': rmse}


def infer_dataset(distortion_types):
    """Infer dataset type from distortion names."""
    d_str = ' '.join(distortion_types).upper()
    if any(k in d_str for k in ['H.264', 'H.265', 'HEVC']):
        return 'CVIQ'
    elif any(k in d_str for k in ['JP2K', 'GN', 'GB']):
        return 'OIQA'
    else:
        raise ValueError("Cannot infer dataset from distortion types.")


def main():
    # Load predictions and ground truth
    pred, gt = load_predictions(PREDICTIONS_PATH)
    mos_values, distortion_types, image_names = load_mos_and_distortion(TEST_CSV_PATH)

    # Validate alignment
    assert len(pred) == len(gt) == len(mos_values), "Mismatch in prediction, GT, and CSV length."
    assert np.allclose(gt, mos_values, atol=1e-5), "Ground truth mismatch between predictions.pt and des.csv"

    # Infer dataset
    global DATASET
    if DATASET is None:
        DATASET = infer_dataset(distortion_types)
    print(f"Dataset inferred: {DATASET}\n")

    # Print header
    print(f"{'Type':12} | {'Metric':6} | {'Value':6}")
    print("-" * 40)

    # Evaluate per distortion
    per_dist = evaluate_per_distortion(pred, gt, distortion_types, DATASET)

    # Evaluate overall
    overall = evaluate_overall(pred, gt)

    # Save to log
    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
    with open(OUTPUT_LOG, 'w') as f:
        f.write(f"MMIFN Evaluation Results - Dataset: {DATASET}\n")
        f.write("="*60 + "\n")
        for d_type, metrics in per_dist.items():
            f.write(f"{d_type:12} | PLCC: {metrics['PLCC']:.3f}, "
                    f"SRCC: {metrics['SRCC']:.3f}, RMSE: {metrics['RMSE']:.3f}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Overall':12} | PLCC: {overall['PLCC']:.3f}, "
                f"SRCC: {overall['SRCC']:.3f}, RMSE: {overall['RMSE']:.3f}\n")

    print(f"\nResults saved to {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
