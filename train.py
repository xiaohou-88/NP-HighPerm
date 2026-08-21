import copy
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metrics import calculate_metrics


def _loss_from_outputs(outputs, targets_reg, targets_cls):
    loss_reg = torch.nn.MSELoss()(outputs[:, 0], targets_reg)
    loss_cls = torch.nn.BCEWithLogitsLoss()(outputs[:, 1], targets_cls.float())
    return loss_reg + 0.1 * loss_cls


def _evaluate_loader(model, loader, device, desc):
    model.eval()
    total_loss = 0.0
    outputs_all = []
    targets_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=True):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch

            fps = fps.to(device)
            graphs = graphs.to(device)
            seqs = seqs.to(device)
            targets_reg = targets_reg.to(device)
            targets_cls = targets_cls.to(device)
            methods = methods.to(device)

            outputs = model(fps, graphs, seqs, methods)
            loss = _loss_from_outputs(outputs, targets_reg, targets_cls)

            total_loss += loss.item()
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(
                np.column_stack((targets_reg.detach().cpu().numpy(), targets_cls.detach().cpu().numpy()))
            )

    avg_loss = total_loss / len(loader)
    outputs_all = np.vstack(outputs_all)
    targets_all = np.vstack(targets_all)
    metrics = calculate_metrics(outputs_all, targets_all)
    return avg_loss, metrics, outputs_all, targets_all


def _save_detailed_predictions(outputs, targets, output_file):
    cls_prob = 1 / (1 + np.exp(-outputs[:, 1]))
    detailed_results = {
        "true_reg": targets[:, 0],
        "pred_reg": outputs[:, 0],
        "true_cls": targets[:, 1],
        "pred_cls_prob": cls_prob,
        "pred_cls": (cls_prob > 0.5).astype(int),
    }
    pd.DataFrame(detailed_results).to_csv(output_file, index=False)


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
    train_results = []
    val_results = []

    best_val_mse = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_outputs = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", disable=True):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch

            fps = fps.to(device)
            graphs = graphs.to(device)
            seqs = seqs.to(device)
            targets_reg = targets_reg.to(device)
            targets_cls = targets_cls.to(device)
            methods = methods.to(device)

            optimizer.zero_grad()
            outputs = model(fps, graphs, seqs, methods)
            loss = _loss_from_outputs(outputs, targets_reg, targets_cls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_outputs.append(outputs.detach().cpu().numpy())
            train_targets.append(
                np.column_stack((targets_reg.detach().cpu().numpy(), targets_cls.detach().cpu().numpy()))
            )

        train_loss /= len(train_loader)
        train_outputs = np.vstack(train_outputs)
        train_targets = np.vstack(train_targets)
        train_metrics = calculate_metrics(train_outputs, train_targets)
        train_results.append({"epoch": epoch, "loss": train_loss, **train_metrics})

        val_loss, val_metrics, _, _ = _evaluate_loader(
            model, val_loader, device, desc=f"Epoch {epoch}/{num_epochs} [Validation]"
        )
        val_results.append({"epoch": epoch, "loss": val_loss, **val_metrics})

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train MSE: {train_metrics['mse']:.4f}, Val MSE: {val_metrics['mse']:.4f}, "
            f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}, "
            f"Train CI: {train_metrics['ci']:.4f}, Val CI: {val_metrics['ci']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
            print(f"New best validation MSE: {best_val_mse:.6f}")
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch} with validation MSE {best_val_mse:.6f}."
            )
            break

    pd.DataFrame(train_results).to_csv(os.path.join(result_dir, "train_results.csv"), index=False)
    pd.DataFrame(val_results).to_csv(os.path.join(result_dir, "validation_results.csv"), index=False)

    if best_model_state is None:
        raise RuntimeError("No best model was saved during training.")

    model.load_state_dict(best_model_state)
    test_loss, test_metrics, test_outputs, test_targets = _evaluate_loader(
        model, test_loader, device, desc="Final Test"
    )

    run_info = {
        "best_epoch": best_epoch,
        "epochs_ran": len(train_results),
        "best_val_mse": best_val_mse,
        "test_loss": test_loss,
    }
    pd.DataFrame([{**run_info, **test_metrics}]).to_csv(
        os.path.join(result_dir, "test_results.csv"), index=False
    )
    _save_detailed_predictions(
        test_outputs,
        test_targets,
        os.path.join(result_dir, "detailed_predictions.csv"),
    )

    return model, test_metrics, run_info


def evaluate(model, test_loader, device):
    _, metrics, _, _ = _evaluate_loader(model, test_loader, device, desc="Testing")
    return metrics
