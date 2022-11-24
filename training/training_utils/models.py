import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, num_features: int, hidden_layer_num_parameters=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_layer_num_parameters),
            nn.ReLU(),
            nn.Linear(hidden_layer_num_parameters,
                      hidden_layer_num_parameters),
            nn.ReLU(),
            nn.Linear(hidden_layer_num_parameters,
                      int(hidden_layer_num_parameters/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layer_num_parameters/2), 1)
        )

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)
