import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, num_features: int, config: dict):
        fc_layer_size = config.get("fc_layer_size", 64)
        hidden_layers = config.get("hidden_layers", 5)
        with_final_half_layer = config.get("with_final_half_layer", True)
        with_relu = config.get("with_relu", True)
        with_dropout = config.get("with_dropout", True)
        dropout = config.get("dropout", 0.3)
        with_softmax = config.get("with_softmax", True)
        modules = []

        super().__init__()

        # input layer
        modules.append(nn.Flatten())
        modules.append(nn.Linear(num_features, fc_layer_size))
        if with_relu:
            modules.append(nn.ReLU())
        if with_dropout:
            modules.append(nn.Dropout(dropout))

        # hidden layers
        for _ in range(hidden_layers):
            modules.append(
                nn.Linear(fc_layer_size, fc_layer_size))
            if with_relu:
                modules.append(nn.ReLU())
            if with_dropout:
                modules.append(nn.Dropout(dropout))

        if with_final_half_layer:
            modules.append(nn.Linear(fc_layer_size,
                                     int(fc_layer_size/2)))
            if with_relu:
                modules.append(nn.ReLU())
            if with_dropout:
                modules.append(nn.Dropout(dropout))

        # last layer
        modules.append(nn.Linear(int(fc_layer_size/2), 1))
        if with_softmax:
            modules.append(nn.LogSoftmax(dim=1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)
