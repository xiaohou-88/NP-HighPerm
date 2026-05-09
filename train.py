import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from metrics import get_cindex, get_rm2
import copy
import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from metrics import calculate_metrics

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def train(model, train_loader, test_loader, optimizer, scheduler, num_epochs, device, result_dir):

    train_results = []
    test_results = []
    
    best_test_mse = float('inf')
    best_model_state = None
    best_metrics = None
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        train_outputs = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch
            
            fps = fps.to(device)
            seqs = seqs.to(device)
            targets_reg = targets_reg.to(device)
            targets_cls = targets_cls.to(device)

            graphs = graphs.to(device)
            methods = methods.to(device)

            optimizer.zero_grad()
            outputs = model(fps, graphs, seqs, methods)

            loss_reg = torch.nn.MSELoss()(outputs[:, 0], targets_reg)
            loss_cls = torch.nn.BCEWithLogitsLoss()(outputs[:, 1], targets_cls.float())
            loss = loss_reg + 0.1 * loss_cls  
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_outputs.append(outputs.detach().cpu().numpy())
            train_targets.append(np.column_stack((targets_reg.cpu().numpy(), targets_cls.cpu().numpy())))
        
        train_loss /= len(train_loader)
        train_outputs = np.vstack(train_outputs)
        train_targets = np.vstack(train_targets)
        
        train_metrics = calculate_metrics(train_outputs, train_targets)

        train_epoch_result = {
            'epoch': epoch + 1,
            'loss': train_loss,
            **train_metrics
        }
        train_results.append(train_epoch_result)
        
        model.eval()
        test_loss = 0.0
        test_outputs = []
        test_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
                fps, graphs, seqs, targets_reg, targets_cls, methods = batch

                fps = fps.to(device)
                seqs = seqs.to(device)
                targets_reg = targets_reg.to(device)
                targets_cls = targets_cls.to(device)
                graphs = graphs.to(device)
                methods = methods.to(device)
                
                outputs = model(fps, graphs, seqs, methods)

                loss_reg = torch.nn.MSELoss()(outputs[:, 0], targets_reg)
                loss_cls = torch.nn.BCEWithLogitsLoss()(outputs[:, 1], targets_cls.float())
                loss = loss_reg + 0.1 * loss_cls
                 
                test_loss += loss.item()
                test_outputs.append(outputs.detach().cpu().numpy())
                test_targets.append(np.column_stack((targets_reg.cpu().numpy(), targets_cls.cpu().numpy())))
        
        test_loss /= len(test_loader)
        test_outputs = np.vstack(test_outputs)
        test_targets = np.vstack(test_targets)
        
        test_metrics = calculate_metrics(test_outputs, test_targets)
        
        test_epoch_result = {
            'epoch': epoch + 1,
            'loss': test_loss,
            **test_metrics
        }
        test_results.append(test_epoch_result)
        
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Train MSE: {train_metrics['mse']:.4f}, "
              f"Test MSE: {test_metrics['mse']:.4f}, "
              f"Train RMSE: {train_metrics['rmse']:.4f}, "
              f"Test RMSE: {test_metrics['rmse']:.4f}, "
              f"Train CI: {train_metrics['ci']:.4f}, "
              f"Test CI: {test_metrics['ci']:.4f}")
        
        if test_metrics['mse'] < best_test_mse:
            best_test_mse = test_metrics['mse']
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = test_metrics
            print(f"发现新的最佳MSE: {best_test_mse:.6f}")
            
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))

    pd.DataFrame(train_results).to_csv(os.path.join(result_dir, 'train_results.csv'), index=False)
    pd.DataFrame(test_results).to_csv(os.path.join(result_dir, 'test_results.csv'), index=False)
    
    detailed_results = {
        'true_reg': test_targets[:, 0],
        'pred_reg': test_outputs[:, 0],
        'true_cls': test_targets[:, 1],
        'pred_cls_prob': 1 / (1 + np.exp(-test_outputs[:, 1])),
        'pred_cls': (1 / (1 + np.exp(-test_outputs[:, 1])) > 0.5).astype(int)
    }
    pd.DataFrame(detailed_results).to_csv(os.path.join(result_dir, 'detailed_predictions.csv'), index=False)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_metrics

def evaluate(model, test_loader, device):

    model.eval()
    test_outputs = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            fps, graphs, seqs, targets_reg, targets_cls, methods = batch

            fps = fps.to(device)
            seqs = seqs.to(device)
            targets_reg = targets_reg.to(device)
            targets_cls = targets_cls.to(device)
            methods = methods.to(device)

            graphs = graphs.to(device)

            outputs = model(fps, graphs, seqs, methods)

            test_outputs.append(outputs.detach().cpu().numpy())
            test_targets.append(np.column_stack((targets_reg.cpu().numpy(), targets_cls.cpu().numpy())))
    
    test_outputs = np.vstack(test_outputs)
    test_targets = np.vstack(test_targets)
    
    metrics = calculate_metrics(test_outputs, test_targets)

    result_dirs = [os.path.join(os.path.dirname(__file__), "new_result", d) for d in os.listdir(os.path.join(os.path.dirname(__file__), "new_result"))]
    if result_dirs:
        latest_dir = max(result_dirs, key=os.path.getctime)

        test_results_df = pd.DataFrame([metrics])
        test_results_df.to_csv(os.path.join(latest_dir, 'test_results.csv'), index=False)

        detailed_results = {
            'true_reg': test_targets[:, 0],
            'pred_reg': test_outputs[:, 0],
            'true_cls': test_targets[:, 1],
            'pred_cls_prob': 1 / (1 + np.exp(-test_outputs[:, 1])),
            'pred_cls': (1 / (1 + np.exp(-test_outputs[:, 1])) > 0.5).astype(int)
        }
        pd.DataFrame(detailed_results).to_csv(os.path.join(latest_dir, 'detailed_predictions.csv'), index=False)
    
    return metrics
