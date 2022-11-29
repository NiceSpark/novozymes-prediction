import torch
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
        modules.append(nn.Dropout(dropout))

        # hidden layers
        for _ in range(hidden_layers):
            modules.append(
                nn.Linear(fc_layer_size, fc_layer_size))
            if with_relu:
                modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))

        if with_final_half_layer:
            modules.append(nn.Linear(fc_layer_size,
                                     int(fc_layer_size/2)))
            if with_relu:
                modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            # last layer
            modules.append(nn.Linear(int(fc_layer_size/2), 1))
        else:
            # last layer
            modules.append(nn.Linear(int(fc_layer_size), 1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


class ThermoNet2(nn.Module):
    def __init__(self, config):
        super().__init__()

        CONV_LAYER_SIZES = [14, 16, 24, 32, 48, 78, 128]
        FLATTEN_SIZES = [0, 5488, 5184, 4000, 3072, 2106, 1024]

        dropout_cnn = config["dropout_cnn"]
        dense_layer_size = int(config["dense_layer_size"])
        layer_num = int(config["conv_layer_num"])

        self.config = config
        activation = nn.ReLU()

        model = [
            nn.Sequential(
                *[nn.Sequential(
                    nn.Conv3d(
                        in_channels=CONV_LAYER_SIZES[l], out_channels=CONV_LAYER_SIZES[l + 1], kernel_size=(3, 3, 3)),
                    activation
                ) for l in range(layer_num)]
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Flatten(),
        ]
        flatten_size = FLATTEN_SIZES[layer_num]
        self.model = nn.Sequential(*model)

        self.ddG = nn.Sequential(
            nn.Dropout(p=dropout_cnn),
            nn.Linear(in_features=flatten_size,
                      out_features=dense_layer_size),
            activation,
            nn.Dropout(p=dropout_cnn),
            nn.Linear(in_features=dense_layer_size, out_features=1)
        )

    def forward(self, x):
        if self.config["diff_features"]:
            x[:, 7:, ...] -= x[:, :7, ...]
        x = self.model(x)
        ddg = self.ddG(x)
        return ddg


class HybridNN(nn.Module):
    def __init__(self, num_features: int, config):
        super().__init__()
        self.diff_features = config["diff_features"]
        self.model_type = config["model_type"]

        # get parameters
        # cnn specifics:
        CONV_LAYER_SIZES = [14, 16, 24, 32, 48, 78, 128]
        FLATTEN_SIZES = [0, 5488, 5184, 4000, 3072, 2106, 1024]
        cnn_dropout = config.get("cnn_dropout", 0.05)
        cnn_dense_layer_size = int(config.get("cnn_dense_layer_size", 64))
        cnn_layer_num = int(config.get("cnn_layer_num", 5))
        cnn_flatten_size = FLATTEN_SIZES[cnn_layer_num]

        # fc specifics:
        regression_dropout = config.get("regression_dropout", 0.3)
        regression_fc_layer_size = config.get("regression_fc_layer_size", 64)
        regression_hidden_layers = config.get("regression_hidden_layers", 5)
        with_final_half_layer = config.get("with_final_half_layer", True)
        with_relu = config.get("with_relu", True)

        cnn_model = []
        regression_model = []

        if self.model_type in ["hybrid", "cnn_only"]:
            ##### build cnn model #####
            cnn_model = [
                nn.Sequential(
                    *[nn.Sequential(
                        nn.Conv3d(
                            in_channels=CONV_LAYER_SIZES[l], out_channels=CONV_LAYER_SIZES[l + 1], kernel_size=(3, 3, 3)),
                        nn.ReLU()
                    ) for l in range(cnn_layer_num)]
                ),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Flatten(),
                nn.Sequential(
                    nn.Dropout(p=cnn_dropout),
                    nn.Linear(in_features=cnn_flatten_size,
                              out_features=cnn_dense_layer_size),
                    nn.ReLU(),
                    nn.Dropout(p=cnn_dropout),
                )
            ]

        if self.model_type in ["hybrid", "regression_only"]:
            ##### build regression model #####
            if self.model_type == "hybrid":
                regression_input_size = cnn_dense_layer_size+num_features
            else:
                # regression only
                regression_input_size = num_features
            # input layer
            regression_model.append(nn.Flatten())
            regression_model.append(nn.Linear(regression_input_size,
                                              regression_fc_layer_size))
            if with_relu:
                regression_model.append(nn.ReLU())
            regression_model.append(nn.Dropout(regression_dropout))

            # hidden layers
            for _ in range(regression_hidden_layers):
                regression_model.append(
                    nn.Linear(regression_fc_layer_size, regression_fc_layer_size))
                if with_relu:
                    regression_model.append(nn.ReLU())
                regression_model.append(nn.Dropout(regression_dropout))

            if with_final_half_layer:
                regression_model.append(nn.Linear(regression_fc_layer_size,
                                                  int(regression_fc_layer_size/2)))
                if with_relu:
                    regression_model.append(nn.ReLU())
                regression_model.append(nn.Dropout(regression_dropout))
                # last layer
                regression_model.append(
                    nn.Linear(int(regression_fc_layer_size/2), 1))
            else:
                # last layer
                regression_model.append(
                    nn.Linear(int(regression_fc_layer_size), 1))
        else:
            # if model_type is cnn_only, then we want a final linear layer
            regression_model.append(
                nn.Linear(in_features=cnn_dense_layer_size, out_features=1))

        self.cnn_model = nn.Sequential(*cnn_model)
        self.regression_model = nn.Sequential(*regression_model)

    def forward(self, voxel_features, features):

        # 1. if we are using the cnn part of the model we apply the cnn to the voxels
        if self.model_type in ["hybrid", "cnn_only"]:
            if self.diff_features:
                voxel_features[:, 7:, ...] -= voxel_features[:, :7, ...]
            cnn_result = self.cnn_model(voxel_features)

        if self.model_type == "hybrid":
            # 2.a if we are in hybrid mode we concat the cnn_result and the other features
            x = torch.cat((cnn_result, features), dim=1)
        elif self.model_type == "cnn_only":
            # 2.b otherwise if we are in cnn only mode we do a final last layer on cnn_result only
            x = cnn_result
        else:
            # 2.c the last case is == "regression", in that case we forget about voxels completly
            x = features

        x = self.regression_model(x)
        return x
