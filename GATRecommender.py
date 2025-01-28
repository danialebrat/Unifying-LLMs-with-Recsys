
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, initial_node_features=None):
        super(GATRecommender, self).__init__()

        self.node_embeddings = nn.Parameter(initial_node_features.clone())
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # First layer
        self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, dropout=0.2, edge_dim=1))
        self.layer_norms.append(nn.LayerNorm(hidden_dim * 4))
        if input_dim != hidden_dim * 4:
            self.residual_projections.append(nn.Linear(input_dim, hidden_dim * 4))
        else:
            self.residual_projections.append(nn.Identity())

        in_channels = hidden_dim * 4

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(in_channels, hidden_dim, heads=4, concat=True, dropout=0.2, edge_dim=1))
            self.layer_norms.append(nn.LayerNorm(hidden_dim * 4))
            if in_channels != hidden_dim * 4:
                self.residual_projections.append(nn.Linear(in_channels, hidden_dim * 4))
            else:
                self.residual_projections.append(nn.Identity())
            in_channels = hidden_dim * 4

        # Last layer
        self.layers.append(GATv2Conv(in_channels, hidden_dim, heads=1, concat=False, dropout=0.2, edge_dim=1))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        if in_channels != hidden_dim:
            self.residual_projections.append(nn.Linear(in_channels, hidden_dim))
        else:
            self.residual_projections.append(nn.Identity())

    def forward(self, data):
        x = self.node_embeddings
        edge_index, edge_attr = data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            x_residual = x
            x = layer(x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.layer_norms[i](x)
            x_residual = self.residual_projections[i](x_residual)
            x = x + x_residual
        return x


# -------------------------------------------------------------------------

