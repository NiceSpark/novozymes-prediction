import torch.nn as nn


class SimpleNN_1(nn.Module):
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


class SimpleNN(nn.Module):
    def __init__(self, num_features: int, model_config: dict):
        nn_linear_parameters = model_config.get("nn_linear_parameters", 64)
        num_hidden_layers = model_config.get("num_hidden_layers", 5)
        with_final_half_layer = model_config.get("with_final_half_layer", True)
        with_relu = model_config.get("with_relu", True)
        super().__init__()
        modules = []
        # input layer
        modules.append(nn.Linear(num_features, nn_linear_parameters))
        if with_relu:
            modules.append(nn.ReLU())
        # hidden layers
        for _ in range(num_hidden_layers):
            modules.append(
                nn.Linear(nn_linear_parameters, nn_linear_parameters))
            if with_relu:
                modules.append(nn.ReLU())

        if with_final_half_layer:
            modules.append(nn.Linear(nn_linear_parameters,
                                     int(nn_linear_parameters/2)))
            if with_relu:
                modules.append(nn.ReLU())

        # last layer
        modules.append(nn.Linear(int(nn_linear_parameters/2), 1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)
