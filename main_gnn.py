import yaml
import torch
import warnings
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data_loader import ADMETDataLoader
from src.graph_features import GraphExtractor
from src.models.gnn import ADMETGraphNet
from src.models.gnn_trainer import GNNTrainer

# Suppress minor RDKit/PyG warnings for clean terminal output
warnings.filterwarnings('ignore')

def prepare_graph_dataset(df, smiles_col, target_col, extractor):
    """Converts a Pandas DataFrame into a list of PyTorch Geometric Data objects."""
    data_list = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[target_col]
        
        # Extract graph
        graph_data = extractor.extract(smiles)
        if graph_data is not None:
            # Attach the ground truth label to the graph
            graph_data.y = torch.tensor([label], dtype=torch.float)
            data_list.append(graph_data)
            
    return data_list

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}\n")

    with open('configs/endpoints.yaml', 'r') as f:
        config = yaml.safe_load(f)

    extractor = GraphExtractor()
    # 9 atoms + 7 degrees + 1 aromatic = 17 node features
    num_node_features = 17 
    
    results = {}

    for endpoint, endpoint_config in config['endpoints'].items():
        print(f"[{endpoint}] Initializing PyTorch Geometric Pipeline...")
        
        task_type = endpoint_config['task_type']
        target_col = endpoint_config['data'].get('target_column', 'Y')
        smiles_col = 'Drug'
        
        # 1. Load Data
        loader = ADMETDataLoader(config, endpoint)
        train_df, valid_df, test_df = loader.get_data_splits()
        
        # 2. Convert to Graph Tensors
        print(f"   Building Graph Tensors...")
        train_dataset = prepare_graph_dataset(train_df, smiles_col, target_col, extractor)
        valid_dataset = prepare_graph_dataset(valid_df, smiles_col, target_col, extractor)
        test_dataset = prepare_graph_dataset(test_df, smiles_col, target_col, extractor)
        
        # 3. Create PyG DataLoaders (Mini-batching)
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 4. Initialize GNN & Trainer
        model = ADMETGraphNet(num_node_features=num_node_features, hidden_channels=64, task_type=task_type)
        trainer = GNNTrainer(model, lr=0.001, task_type=task_type, device='cuda')
        # 5. Training Loop
        epochs = 30
        print(f"   Training Graph Neural Network for {epochs} Epochs...")
        
        # Using a simple progress bar
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            loss = trainer.train_epoch(train_loader)
            
        # 6. Evaluate on Test Set
        print(f"   Evaluating on Scaffold Split Test Set...")
        test_metrics = trainer.evaluate(test_loader)
        results[endpoint] = test_metrics
        print("-" * 50)

    # 7. Print Final Leaderboard
    print("\n==================================================")
    print(" GNN MODULAR EVALUATION REPORT ")
    print("==================================================")
    print(f"{'Endpoint':>20} | {'ROC-AUC':>8} | {'RMSE':>8}")
    print("-" * 50)
    
    for endpoint, metrics in results.items():
        auc = f"{metrics.get('roc_auc', 0):.3f}" if 'roc_auc' in metrics else "   -    "
        rmse = f"{metrics.get('rmse', 0):.3f}" if 'rmse' in metrics else "   -    "
        print(f"{endpoint:>20} | {auc:>8} | {rmse:>8}")
    print("==================================================")

if __name__ == "__main__":
    main()