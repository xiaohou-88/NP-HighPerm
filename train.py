import copy
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metrics import calculate_metrics




def _evaluate_loader(model, loader, device, desc):
   

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=True):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch

            

  
    return avg_loss, metrics, outputs_all, targets_all


def _save_detailed_predictions(outputs, targets, output_file):
   


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    result_dir,
    early_stopping_patience=20,
):
    

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_outputs = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", disable=True):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch

           

    pd.DataFrame(train_results).to_csv(os.path.join(result_dir, "train_results.csv"), index=False)
    pd.DataFrame(val_results).to_csv(os.path.join(result_dir, "validation_results.csv"), index=False)

    

    return model, test_metrics, run_info


def evaluate(model, test_loader, device):
    _, metrics, _, _ = _evaluate_loader(model, test_loader, device, desc="Testing")
    return metrics
