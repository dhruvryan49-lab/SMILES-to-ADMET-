import optuna
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import warnings

from src.graph_features import GraphExtractor
from src.models.gnn import ADMETGraphNet
from src.models.gnn_trainer import GNNTrainer

warnings.filterwarnings('ignore')

def prepare_graph_dataset(df, extractor):
    data_list = []
    for _, row in df.iterrows():
        graph_data = extractor.extract(row['Drug'])
        if graph_data is not None and graph_data.edge_index.numel() > 0:
            graph_data.y = torch.tensor([row['Y']], dtype=torch.float)
            data_list.append(graph_data)
    return data_list

def objective(trial):
    # 1. Optuna Hyperparameter Space
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    heads = trial.suggest_int('heads', 2, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # 2. PyG Mini-batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 3. Build Model with Trial Parameters
    model = ADMETGraphNet(
        num_node_features=17, 
        hidden_channels=hidden_channels, 
        task_type='regression',
        dropout=dropout,
        heads=heads
    )
    trainer = GNNTrainer(model, lr=lr, task_type='regression', device='cuda')

    # 4. Fast Training Loop (20 epochs for rapid search)
    epochs = 20
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(train_loader)

    # 5. Evaluate on Validation Set
    metrics = trainer.evaluate(valid_loader)
    
    # Optuna needs to know if the trial was good or bad
    return metrics['rmse']

if __name__ == "__main__":
    print("[EGFR] Loading and Pre-extracting Graphs for Optuna...")
    df = pd.read_csv('data/chembl_egfr.csv')
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)

    extractor = GraphExtractor()
    
    # Extract once, hold in RAM globally
    global train_dataset, valid_dataset
    train_dataset = prepare_graph_dataset(train_df, extractor)
    valid_dataset = prepare_graph_dataset(valid_df, extractor)
    
    print("\nStarting Optuna Hyperparameter Search (15 Trials)...")
    study = optuna.create_study(direction='minimize')
    # Limit to 15 trials so it finishes in a reasonable time on the RTX 5050
    study.optimize(objective, n_trials=15)

    print("\n==================================================")
    print(" GATv2 OPTIMIZATION COMPLETE ")
    print("==================================================")
    print(f" Best Validation RMSE: {study.best_value:.3f}")
    print(" Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print("==================================================")