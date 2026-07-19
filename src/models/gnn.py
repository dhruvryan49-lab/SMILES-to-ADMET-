import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class ADMETGraphNet(torch.nn.Module):
    # Added dropout and heads to the arguments
    def __init__(self, num_node_features, hidden_channels, task_type='classification', dropout=0.5, heads=3):
        super(ADMETGraphNet, self).__init__()
        self.task_type = task_type
        self.dropout = dropout # Store dropout dynamically
        
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, edge_dim=2, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=2, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=2, heads=heads, concat=False)
        
        self.lin = Linear(hidden_channels + 8, 17)

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        
        x = global_mean_pool(x, batch)
        x = torch.cat([x, global_features], dim=1)
        
        # Use the dynamic dropout value here
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin(x)
        
        if self.task_type == 'classification':
            return torch.sigmoid(out)
        return out