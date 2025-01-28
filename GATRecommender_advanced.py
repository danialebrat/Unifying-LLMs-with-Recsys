import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATBlock(nn.Module):
    """
    A single GAT block that:
      - applies GATv2Conv
      - applies ELU (or any activation)
      - optionally includes a feed-forward network
      - has gated residual connections
      - normalizes the output
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=4,
                 dropout=0.2,
                 edge_dim=1,
                 use_ffn=True,
                 ffn_hidden_factor=2,
                 normalization='layernorm'):
        super(GATBlock, self).__init__()

        self.gat_conv = GATv2Conv(in_channels, out_channels, heads=heads,
                                  concat=True if heads > 1 else False,
                                  dropout=dropout,
                                  edge_dim=edge_dim)

        effective_out = out_channels * heads if heads > 1 else out_channels

        # Gated Residual parameters
        # We will project the input to match the output if necessary
        self.proj = (nn.Linear(in_channels, effective_out)
                     if in_channels != effective_out
                     else nn.Identity())
        self.gate = nn.Linear(effective_out, 1)  # For a simple gate over the entire embedding

        # Optional feed-forward sub-layer
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(effective_out, ffn_hidden_factor * effective_out),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden_factor * effective_out, effective_out),
            )

        # Normalization
        if normalization.lower() == 'layernorm':
            self.norm = nn.LayerNorm(effective_out)
        elif normalization.lower() == 'batchnorm':
            self.norm = nn.BatchNorm1d(effective_out)
        else:
            raise ValueError("normalization must be 'layernorm' or 'batchnorm'")

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # GAT Forward
        x_res = x
        x = self.gat_conv(x, edge_index, edge_attr)
        x = self.activation(x)

        # Gated Residual
        x_proj = self.proj(x_res)
        gate_val = torch.sigmoid(self.gate(x))
        x = gate_val * x + (1 - gate_val) * x_proj

        # Optional feed-forward
        if self.use_ffn:
            x_ffn = self.ffn(x)
            # Another residual over FFN
            x = x + x_ffn

        # Normalize
        # For batch norm, we need x to be [batch_size, feature_dim]
        # which typically implies x is shaped as [num_nodes, feature_dim].
        # If x is [num_nodes, feature_dim], we can just pass x through.
        # But if it’s [batch_size, ..., feature_dim], we’d reshape accordingly.
        x = self.norm(x)

        # Additional dropout (if desired)
        x = self.dropout(x)

        return x


class AdvancedGATRecommender(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers=3,
                 initial_node_features=None,
                 heads_per_layer=None,
                 dropout_schedule=None,
                 normalization='layernorm'):
        """
        :param input_dim: Input dimensionality for each node.
        :param hidden_dim: Base hidden dimension for GAT layers.
        :param num_layers: How many GAT blocks to stack.
        :param initial_node_features: Torch tensor of shape [num_nodes, input_dim].
        :param heads_per_layer: List or tuple specifying heads for each layer, length = num_layers.
                                If None, defaults to [4, 4, 1] for example.
        :param dropout_schedule: List or tuple for dropout rates for each layer (length = num_layers).
                                 If None, defaults to [0.2, 0.2, 0.2].
        :param normalization: 'layernorm' or 'batchnorm'.
        """
        super(AdvancedGATRecommender, self).__init__()

        # Initialize node embeddings
        self.node_embeddings = nn.Parameter(initial_node_features.clone())

        if heads_per_layer is None:
            # e.g., first two layers use 4 heads, last layer uses single head
            heads_per_layer = [4] * (num_layers - 1) + [1]
        if dropout_schedule is None:
            dropout_schedule = [0.2] * num_layers

        # Build the stack of GAT blocks
        self.layers = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers
                out_dim = hidden_dim
            else:
                # Last layer
                out_dim = hidden_dim  # or any final dimension you desire

            self.layers.append(
                GATBlock(in_channels=in_dim,
                         out_channels=out_dim,
                         heads=heads_per_layer[i],
                         dropout=dropout_schedule[i],
                         edge_dim=1,
                         use_ffn=True,  # set True or False as you prefer
                         ffn_hidden_factor=2,  # can tune
                         normalization=normalization)
            )

            # If heads > 1, the effective out_dim = out_dim * heads
            # So for next layer:
            if heads_per_layer[i] > 1:
                in_dim = out_dim * heads_per_layer[i]
            else:
                in_dim = out_dim

    def forward(self, data):
        x = self.node_embeddings
        edge_index, edge_attr = data.edge_index, data.edge_attr

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x
