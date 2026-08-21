import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from feature_engineering import MolDataset, collate_fn
from model import MultiModalNet
from train import train


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
METRIC_COLUMNS = [
    "mse",
    "rmse",
    "r2",
    "ci",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "consistency",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_permeability_bins(values, max_bins=10):
    values = pd.Series(values)
    for bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if labels.nunique(dropna=True) < 2:
            continue
        counts = labels.value_counts(dropna=False)
        if counts.min() >= 2:
            return labels
    return None


def split_train_validation(train_file, fold_dir, val_ratio, seed):
    df = pd.read_csv(train_file)
    bins = make_permeability_bins(df["Standardized_Value"])
    stratify = bins if bins is not None else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    train_split_file = os.path.join(fold_dir, "train_split.csv")
    val_split_file = os.path.join(fold_dir, "validation_split.csv")
    train_df.to_csv(train_split_file, index=False)
    val_df.to_csv(val_split_file, index=False)

    split_info = {
        "source_train_rows": len(df),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "validation_ratio": len(val_df) / len(df),
        "stratified_by_standardized_value": stratify is not None,
        "train_value_mean": train_df["Standardized_Value"].mean(),
        "validation_value_mean": val_df["Standardized_Value"].mean(),
        "train_value_std": train_df["Standardized_Value"].std(),
        "validation_value_std": val_df["Standardized_Value"].std(),
        "train_value_min": train_df["Standardized_Value"].min(),
        "validation_value_min": val_df["Standardized_Value"].min(),
        "train_value_max": train_df["Standardized_Value"].max(),
        "validation_value_max": val_df["Standardized_Value"].max(),
    }
    pd.DataFrame([split_info]).to_csv(os.path.join(fold_dir, "split_info.csv"), index=False)
    return train_split_file, val_split_file, split_info


def run_cross_validation(
    folds_dir,
    device,
    batch_size,
    num_epochs,
    seed,
    val_ratio=0.1,
    early_stopping_patience=20,
):
    set_seed(seed)

    main_result_dir = os.path.join(os.path.dirname(__file__), "new_result", timestamp + "_trainval_crossval")
    os.makedirs(main_result_dir, exist_ok=True)

    all_metrics = []
    fold_summaries = []

    for fold in range(1, 6):
        print(f"\n=== Start fold {fold} training ===")
        fold_seed = seed + fold
        set_seed(fold_seed)

        fold_dir = os.path.join(main_result_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_file = os.path.join(folds_dir, f"fold_{fold}_train.csv")
        test_file = os.path.join(folds_dir, f"fold_{fold}_test.csv")
        train_split_file, val_split_file, split_info = split_train_validation(
            train_file=train_file,
            fold_dir=fold_dir,
            val_ratio=val_ratio,
            seed=fold_seed,
        )

        train_set = MolDataset(train_split_file)
        val_set = MolDataset(val_split_file, vocab=train_set.vocab, word2id=train_set.word2id)
        test_set = MolDataset(test_file, vocab=train_set.vocab, word2id=train_set.word2id)

        print(
            f"Fold {fold} sizes - train: {len(train_set)}, "
            f"validation: {len(val_set)}, test: {len(test_set)}"
        )

        train_generator = torch.Generator()
        train_generator.manual_seed(fold_seed)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=train_generator,
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = MultiModalNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        _, test_metrics, run_info = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            result_dir=fold_dir,
            early_stopping_patience=early_stopping_patience,
        )

        all_metrics.append(test_metrics)
        fold_summary = {"fold": fold, **split_info, **run_info, **test_metrics}
        fold_summaries.append(fold_summary)
        pd.DataFrame([fold_summary]).to_csv(os.path.join(fold_dir, f"fold_{fold}_test_results.csv"), index=False)
        print(f"Fold {fold} final test results: {test_metrics}")

    summary_metrics = {}
    for metric in METRIC_COLUMNS:
        values = np.array(
            [fold_metric[metric] for fold_metric in all_metrics],
            dtype=float,
        )
        summary_metrics[f"{metric}_mean"] = np.mean(values)
        summary_metrics[f"{metric}_std"] = np.std(values, ddof=1)

    pd.DataFrame(fold_summaries).to_csv(os.path.join(main_result_dir, "fold_summary.csv"), index=False)
    pd.DataFrame([summary_metrics]).to_csv(os.path.join(main_result_dir, "average_results.csv"), index=False)

    print("\n=== 5-fold final test mean +/- SD ===")
    for metric in METRIC_COLUMNS:
        mean = summary_metrics[f"{metric}_mean"]
        std = summary_metrics[f"{metric}_std"]
        print(f"{metric}: {mean:.4f} +/- {std:.4f}")

    return summary_metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    folds_dir = os.path.join(os.path.dirname(__file__), "all_data_split2", "folds")
    run_cross_validation(
        folds_dir=folds_dir,
        device=device,
        batch_size=128,
        num_epochs=100,
        seed=42,
        val_ratio=0.1,
        early_stopping_patience=20,
    )
