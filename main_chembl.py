import pandas as pd
import torch
import warnings
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.graph_features import GraphExtractor
from src.models.gnn import ADMETGraphNet
from src.models.gnn_trainer import GNNTrainer

warnings.filterwarnings('ignore')

def prepare_graph_dataset(df, extractor):
    data_list = []
    # Drop any rows where RDKit fails to parse the SMILES
    for _, row in df.iterrows():
        graph_data = extractor.extract(row['Drug'])
        if graph_data is not None and graph_data.edge_index.numel() > 0:
            graph_data.y = torch.tensor([row['Y']], dtype=torch.float)
            data_list.append(graph_data)
    return data_list

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}\n")

    print("[EGFR] Loading ChEMBL Dataset...")
    df = pd.read_csv('data/chembl_egfr.csv')
    
    # 80/10/10 Train/Valid/Test Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    extractor = GraphExtractor()
    num_node_features = 17 
    
    print("Building Graph Tensors (This will take a minute for 10k molecules)...")
    train_dataset = prepare_graph_dataset(train_df, extractor)
    valid_dataset = prepare_graph_dataset(valid_df, extractor)
    test_dataset = prepare_graph_dataset(test_df, extractor)
    
    print(f"   Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    # Mini-batching for the GPU
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Hybrid GNN with Trial 3 Hyperparameters
    model = ADMETGraphNet(
        num_node_features=17, 
        hidden_channels=64, 
        task_type='regression', 
        dropout=0.23, 
        heads=2
    )
    trainer = GNNTrainer(model, lr=0.0035, task_type='regression', device='cuda')

    epochs = 150
    patience = 15  # How many epochs to wait for an improvement before stopping
    best_valid_rmse = float('inf')
    epochs_without_improvement = 0

    print(f"\nTraining Hybrid GATv2 with Early Stopping (Max {epochs} Epochs)...")
    
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate on validation set to check for overfitting
        valid_metrics = trainer.evaluate(valid_loader)
        current_valid_rmse = valid_metrics['rmse']
        
        # Check if this is the best epoch so far
        if current_valid_rmse < best_valid_rmse:
            best_valid_rmse = current_valid_rmse
            epochs_without_improvement = 0
            # Print only when it finds a new best to keep the terminal clean
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.3f} | Valid RMSE: {current_valid_rmse:.3f} (New Best!)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n[Early Stopping Triggered] Validation score hasn't improved for {patience} epochs.")
                break
                
    print(f"\nEvaluating Final Model on Test Set...")
    # The true test: evaluating on the unseen 10% test split
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n==================================================")
    print(" FINAL OPTIMIZED CHEMBL EGFR REPORT ")
    print("==================================================")
    print(f" Test RMSE: {test_metrics['rmse']:.3f}")
    print(f" Test R-Squared: {test_metrics['r2']:.3f}")
    print("==================================================")

if __name__ == "__main__":
    main()