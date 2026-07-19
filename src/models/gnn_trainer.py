import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import numpy as np

class GNNTrainer:
    def __init__(self, model, lr=0.001, task_type='classification', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.task_type = task_type
        
        if self.task_type == 'classification':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Ensure global features shape matches [batch_size, 8]
            global_feat = batch.global_features.view(batch.num_graphs, -1)
            
            # Forward pass (now passing global_feat)
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, global_feat)
            
            if self.task_type == 'classification':
                loss = self.criterion(out.squeeze(), batch.y.float())
            else:
                loss = self.criterion(out.squeeze(), batch.y.float())
                
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
                
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(out.squeeze().cpu().numpy())
                
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if self.task_type == 'classification':
            auc = roc_auc_score(y_true, y_pred)
            return {'roc_auc': auc}
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return {'rmse': rmse, 'r2': r2}