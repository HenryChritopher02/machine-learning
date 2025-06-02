
import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadp
from utils.model_seed import set_seed

#  set_seed(seed=42)

def create_mlp(input_size, output_size):
    return Sequential(Linear(input_size, output_size, bias=False), ReLU(), Linear(output_size, output_size, bias=False))

class GIN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GIN, self).__init__()

        # Assign hyperparameters
        self.embedding_size = model_params['embedding_size']
        self.dense_neuron = model_params['dense_neuron']
        self.num_gin_layers = model_params['num_gin_layers']
        self.num_mlp_layers = model_params['num_mlp_layers']
        self.dropout_value = model_params['dropout_value']
        self.dropout_mlp = model_params['dropout_mlp']

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

        # MLP Layers after concatenation
        mlp_layers = []
        mlp_input_size = self.embedding_size * 3  # After global pooling, this is the size
        mlp_output_size = self.dense_neuron
        for i in range(self.num_mlp_layers):
            # Define each layer size, starting with self.dense_neuron and halving it each time
            mlp_layers.append(Linear(mlp_input_size, mlp_output_size))
            mlp_layers.append(ReLU())
            mlp_layers.append(torch.nn.Dropout(p=self.dropout_mlp))  # Apply dropout after each MLP layer
            mlp_input_size = mlp_output_size
            mlp_output_size = max(1, mlp_input_size // 2)

        # Register the MLP layers
        self.mlp = torch.nn.Sequential(*mlp_layers)

        # Output layer
        self.out = Linear(mlp_input_size, 1)

    def forward(self, x, edge_index, batch_index):
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

        # MLP layers after concatenation
        hidden = self.mlp(hidden)

        # Output layer
        out = self.out(hidden)

        return out