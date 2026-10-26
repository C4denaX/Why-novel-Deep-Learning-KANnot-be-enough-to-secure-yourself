"""DL and KAN architectures used in the study (Methodology, "DL Architectures").

The output layer of every model has one unit per class of the dataset. The
KAN-specific hyperparameters are fixed (Table 3) and read from the configuration:
ConKAN layers with grid size 5 and spline order 3 on [-1, 1] with GELU base
activation, the KANLinear layer of Conv & KAN with grid size 10 and spline order 3
on [0, 1] with SiLU base activation, and FastKAN with 8 Gaussian radial basis
functions on [-2, 2] with SiLU base activation.
"""
from architectures.ConvNet import ConvNet1D
from architectures.conv_and_kan import NormalConvsKAN
from architectures.kan1d import ConvKAN, Simple_ConvKAN, Simple_ConvKAN_2
from architectures.RnnModels import GRUMalware_Model1, LSTMMalware_Model1, RNNMalware_Model1
from architectures.SimpleModels import SimpleCNN, SimpleCNN_2
from fastkan import FastKAN

# Model name (as in the result files) -> input layout.
# "CNN" and "RNN": (batch, 1, features); "None": (batch, features).
LAYOUT = {
    "CNN (Small)": "CNN",
    "CNN (Medium)": "CNN",
    "CNN (Big)": "CNN",
    "ConvKAN (Small)": "CNN",
    "ConvKAN (Medium)": "CNN",
    "ConvKAN (Big)": "CNN",
    "Convs_and_KAN": "CNN",
    "RNN": "RNN",
    "LSTM": "RNN",
    "GRU": "RNN",
    "KAN": "None",
}


def build_model(name, num_features, num_classes, kan_cfg):
    conkan = kan_cfg["ConKAN"]
    convs_kan = kan_cfg["Conv & KAN"]
    fastkan = kan_cfg["KAN"]
    if name == "CNN (Small)":
        return SimpleCNN(num_features, num_outputs=num_classes)
    if name == "CNN (Medium)":
        return SimpleCNN_2(num_features, num_outputs=num_classes)
    if name == "CNN (Big)":
        return ConvNet1D(num_features, num_outputs=num_classes)
    if name == "ConvKAN (Small)":
        return Simple_ConvKAN(input_size=num_features, num_outputs=num_classes,
                              grid_size=conkan["grid_size"], spline_order=conkan["spline_order"])
    if name == "ConvKAN (Medium)":
        return Simple_ConvKAN_2(input_size=num_features, num_outputs=num_classes,
                                grid_size=conkan["grid_size"], spline_order=conkan["spline_order"])
    if name == "ConvKAN (Big)":
        return ConvKAN(input_size=num_features, num_outputs=num_classes,
                       grid_size=conkan["grid_size"], spline_order=conkan["spline_order"])
    if name == "Convs_and_KAN":
        return NormalConvsKAN(num_features, num_outputs=num_classes,
                              grid_size=convs_kan["grid_size"], spline_order=convs_kan["spline_order"])
    if name == "RNN":
        return RNNMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "LSTM":
        return LSTMMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "GRU":
        return GRUMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "KAN":
        return FastKAN(layers_hidden=[num_features, 128, num_classes], num_grids=fastkan["num_grids"],
                       grid_min=fastkan["grid_min"], grid_max=fastkan["grid_max"])
    raise ValueError(f"unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
