import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from feature_engineering import MolDataset, collate_fn
from model import MultiModalNet
from train import evaluate

def test_model(test_csv_path, model_path, device, batch_size=128):
    print(f"Loading test data from: {test_csv_path}")
    test_set = MolDataset(test_csv_path)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test set size: {len(test_set)}")

    print(f"Loading model weights from: {model_path}")
    model = MultiModalNet().to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print("Starting evaluation...")
    metrics = evaluate(model, test_loader, device)
    
    print("\n=== Test Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TEST_CSV = os.path.join(os.path.dirname(__file__), "all_data_split", "test_data.csv")

    MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), "new_result", "YOUR_TIMESTAMP_crossval", "fold_1", "best_model.pth")

    if os.path.exists(TEST_CSV) and os.path.exists(MODEL_WEIGHTS):
        test_model(TEST_CSV, MODEL_WEIGHTS, device)
    else:
        print("Error: Test CSV file or model weights not found. Please check the paths.")