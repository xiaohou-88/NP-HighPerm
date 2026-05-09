import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
import datetime
from feature_engineering import MolDataset
from model import MultiModalNet
from train import train, evaluate
from metrics import *
from feature_engineering import collate_fn

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
def run_cross_validation(folds_dir, device, batch_size, num_epochs, seed):

    torch.manual_seed(seed)
    np.random.seed(seed)

    main_result_dir = os.path.join(os.path.dirname(__file__), "new_result", timestamp + "_crossval")
    os.makedirs(main_result_dir, exist_ok=True)

    all_metrics = []
    
    for fold in range(1, 6):
        
        
        fold_dir = os.path.join(main_result_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_file = os.path.join(folds_dir, f"fold_{fold}_train.csv")
        test_file = os.path.join(folds_dir, f"fold_{fold}_test.csv")
        
        train_set = MolDataset(train_file)
        test_set = MolDataset(test_file)

        print(f"Training set size: {len(train_set)}, Test set size: {len(test_set)}")

        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
        model = MultiModalNet().to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_model, best_metrics = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            result_dir=fold_dir
        )
        
        all_metrics.append(best_metrics)
        print(f"{fold}fold test results: {best_metrics}")

        pd.DataFrame([best_metrics]).to_csv(os.path.join(fold_dir, f'fold_{fold}_test_results.csv'), index=False)
    
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold_metric[metric] for fold_metric in all_metrics])
    
    pd.DataFrame([avg_metrics]).to_csv(os.path.join(main_result_dir, 'average_results.csv'), index=False)

    print("\n=== Cross-validation average results ===")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    return avg_metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modify to include the path to the 5-fold data
    folds_dir = os.path.join(os.path.dirname(__file__), "all_data_split2", "folds")

    # Run 5-fold cross-validation
    run_cross_validation(
        folds_dir=folds_dir,
        device=device,
        batch_size=128,  
        num_epochs=100,  
        seed=42
    )


