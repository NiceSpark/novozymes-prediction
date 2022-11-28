import torch


class ThermoNet2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        CONV_LAYER_SIZES = [14, 16, 24, 32, 48, 78, 128]
        FLATTEN_SIZES = [0, 5488, 5184, 4000, 3072, 2106, 1024]

        dropout_rate = config['dropout_rate']
        dense_layer_size = int(config['dense_layer_size'])
        layer_num = int(config['conv_layer_num'])

        self.config = config
        activation = torch.nn.ReLU()

        model = [
            torch.nn.Sequential(
                *[torch.nn.Sequential(
                    torch.nn.Conv3d(
                        in_channels=CONV_LAYER_SIZES[l], out_channels=CONV_LAYER_SIZES[l + 1], kernel_size=(3, 3, 3)),
                    activation
                ) for l in range(layer_num)]
            ),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.Flatten(),
        ]
        flatten_size = FLATTEN_SIZES[layer_num]
        self.model = torch.nn.Sequential(*model)

        self.ddG = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features=flatten_size,
                            out_features=dense_layer_size),
            activation,
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features=dense_layer_size, out_features=1)
        )

    def forward(self, x):
        if self.config['diff_features']:
            x[:, 7:, ...] -= x[:, :7, ...]
        x = self.model(x)
        ddg = self.ddG(x)
        return ddg


def load_Thermonet_model(fname, config):
    model = ThermoNet2(config)
    model.load_state_dict(torch.load(fname))
    return model
