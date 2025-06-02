import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadp
from utils.model_seed import set_seed

set_seed(seed=42)

def create_mlp(input_size, output_size):
    return Sequential(Linear(input_size, output_size, bias=False), ReLU(), Linear(output_size, output_size, bias=False))

class GIN_hybrid(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        # Init parent
        super(GIN_hybrid, self).__init__()

        # Assign hyperparameters
        self.embedding_size = model_params['embedding_size']
        self.dense_neuron = model_params['dense_neuron']
        self.num_gin_layers = model_params['num_gin_layers']
        self.dropout_value = model_params['dropout_value']

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=self.dropout_value)

        # BatchNorm
        self.batchnorm = BatchNorm1d(self.embedding_size)

        # GIN Layers
        self.initial_conv = GINConv(create_mlp(feature_size, self.embedding_size))
        self.gin_layers = torch.nn.ModuleList([
            GINConv(create_mlp(self.embedding_size, self.embedding_size), train_eps=False)
            for _ in range(self.num_gin_layers - 1)
        ])
        # self.out = Linear(dense_neuron, 1)

        self.out = Linear(self.embedding_size * 3, self.dense_neuron)

    def forward(self, x, edge_index, batch_index):
        x, edge_index, batch_index = x.to(next(self.parameters()).device), edge_index.to(next(self.parameters()).device), batch_index.to(next(self.parameters()).device)
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        # Subsequent GIN layers
        for gin_layer in self.gin_layers:
            hidden = gin_layer(hidden, edge_index)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([
            gmp(hidden, batch_index),
            gap(hidden, batch_index),
            gadp(hidden, batch_index)
        ], dim=1)

        out = self.out(hidden)

        return out
    
class MLP1(nn.Module):
    def __init__(self, input_dim, model_params):
        super(MLP1, self).__init__()

        # Assign hyperparameters
        self.dense_neuron = model_params['inner_dense_neuron']
        self.num_layers = model_params['inner_num_layers']
        self.dropout_mlp = model_params['inner_dropout_mlp']

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, self.dense_neuron))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(self.dropout_mlp))

        # Hidden layers
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.dense_neuron, self.dense_neuron))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_mlp))

        # Output layer
        self.layers.append(nn.Linear(self.dense_neuron, self.dense_neuron))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class CombinedMLP(nn.Module):
    def __init__(self, input_dim, model_params):
        super(CombinedMLP, self).__init__()

        # Assign hyperparameters
        self.dense_neuron = model_params['outer_dense_neuron']
        self.num_layers = model_params['outer_num_layers']
        self.dropout_mlp = model_params['outer_dropout_mlp']

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, self.dense_neuron))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(self.dropout_mlp))
        
        # Hidden layers
        mlp_input_size = self.dense_neuron
        mlp_output_size = self.dense_neuron // 2
        for i in range(self.num_layers):  # Loop through the number of layers
            self.layers.append(nn.Linear(mlp_input_size, mlp_output_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_mlp))  # Apply dropout after each MLP layer
            mlp_input_size = mlp_output_size
            mlp_output_size = max(1, mlp_input_size // 2)

        # Output layer
        self.layers.append(nn.Linear(mlp_input_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x