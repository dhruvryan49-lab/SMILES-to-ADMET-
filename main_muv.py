import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import warnings
import os

from src.graph_features import GraphExtractor
from src.models.gnn import ADMETGraphNet
from src.models.gnn_trainer import GNNTrainer

warnings.filterwarnings('ignore')

def download_muv():
    data_path = 'data/muv.csv'
    if not os.path.exists(data_path):
        print("Downloading MUV Dataset from AWS S3 (93,000 records)...")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
        df = pd.read_csv(url, compression='gzip')
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
    return pd.read_csv(data_path)

def prepare_multi_task_graphs(df, extractor, task_cols):
    data_list = []
    # Using tqdm in the terminal will help track this massive extraction
    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Graphs"):
        graph = extractor.extract(row['smiles'])
        if graph is not None and graph.edge_index.numel() > 0:
            # Extract all 17 tasks. MUV has missing data (NaNs), we fill with -1 to mask later
            labels = row[task_cols].fillna(-1).values.astype(np.float32)
            graph.y = torch.tensor([labels], dtype=torch.float)
            data_list.append(graph)
    return data_list

# Custom Trainer specifically for handling masked Multi-Task Loss
class MultiTaskTrainer(GNNTrainer):
    def __init__(self, model, lr=0.0035, device='cuda'):
        super().__init__(model, lr=lr, task_type='regression', device=device)
        # BCEWithLogitsLoss is numerically stable for multi-task
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            global_feat = batch.global_features.view(batch.num_graphs, -1)
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, global_feat)
            
            # Mask out missing data (-1) so it doesn't impact gradients
            mask = batch.y != -1
            loss_tensor = self.criterion(out, batch.y)
            loss = (loss_tensor * mask).sum() / mask.sum() # Average loss only over valid labels
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        return total_loss / len(loader.dataset)

    def evaluate(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                global_feat = batch.global_features.view(batch.num_graphs, -1)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, global_feat)
                
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(out.cpu().numpy())
                
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        roc_aucs = []
        # Calculate ROC-AUC for each of the 17 tasks independently
        for i in range(y_true.shape[1]):
            valid_idx = y_true[:, i] != -1
            if valid_idx.sum() > 0 and len(np.unique(y_true[valid_idx, i])) > 1:
                auc = roc_auc_score(y_true[valid_idx, i], y_pred[valid_idx, i])
                roc_aucs.append(auc)
                
        return {'roc_auc': np.mean(roc_aucs) if roc_aucs else 0.0}

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    df = download_muv()
    task_cols = [col for col in df.columns if col.startswith('MUV-')]
    print(f"Loaded {len(df)} molecules across {len(task_cols)} target tasks.")

    # IMPORTANT: 93k molecules takes hours to convert to graphs. 
    # For a quick test to ensure gradients flow, we sample 10,000. 
    # Remove the .sample() for the final overnight run.
    df = df.sample(10000, random_state=42) 
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    extractor = GraphExtractor()
    train_dataset = prepare_multi_task_graphs(train_df, extractor, task_cols)
    valid_dataset = prepare_multi_task_graphs(valid_df, extractor, task_cols)
    test_dataset = prepare_multi_task_graphs(test_df, extractor, task_cols)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ADMETGraphNet(num_node_features=17, hidden_channels=64, task_type='classification', dropout=0.23, heads=2)
    trainer = MultiTaskTrainer(model, lr=0.0035, device='cuda')

    print("\nTraining Hybrid GATv2 on MUV (Multi-Task Learning)...")
    epochs = 50
    best_auc = 0
    
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(valid_loader)
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Valid Mean ROC-AUC: {best_auc:.4f} (New Best!)")

    test_metrics = trainer.evaluate(test_loader)
    print("\n==================================================")
    print(" MUV MULTI-TASK EVALUATION REPORT ")
    print("==================================================")
    print(f" Global Mean Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()