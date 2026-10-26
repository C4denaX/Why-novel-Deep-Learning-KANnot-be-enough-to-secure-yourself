import torch
import torch.nn as nn

from kan_convolutional.kan_conv import KANConv2DLayer, KANConv1DLayer 
import torch.nn.functional as F

class Simple_ConvKAN(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1,
            grid_size: int = 5,
            num_outputs: int = 15,
            ):
        # num_outputs: size of the output layer (15 unless set by the caller).
        # num_classes is accepted for compatibility and does not change the model.
        super(Simple_ConvKAN, self).__init__()

        self.conv1 = KANConv1DLayer(input_dim=input_channels, output_dim=5, spline_order=spline_order, kernel_size=3, groups=1, padding=1, stride=1,
                           dilation=1, grid_size=grid_size)
        self.conv2 = KANConv1DLayer(input_dim=5, output_dim=5, spline_order=spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=1, dilation=1, grid_size=grid_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Calcular la dimensión de entrada para la primera capa totalmente conectada
        self.flatten_dim = self.calculate_flatten_dim(input_size)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flatten_dim, num_outputs)

    def calculate_flatten_dim(self, num_features):
        # Crear un tensor dummy para pasar por la red y obtener el tamaño de la salida
        x = torch.zeros(1, 1, num_features)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class Simple_ConvKAN_2(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1,
            grid_size: int = 5,
            num_outputs: int = 15,
            ):
        # num_outputs: size of the output layer (15 unless set by the caller).
        # num_classes is accepted for compatibility and does not change the model.
        super(Simple_ConvKAN_2, self).__init__()

        self.conv1 = KANConv1DLayer(input_dim=input_channels, output_dim=5, spline_order=spline_order, kernel_size=3, groups=1, padding=0, stride=1,
                           dilation=1, grid_size=grid_size)
        self.conv2 = KANConv1DLayer(input_dim=5, output_dim=25, spline_order=spline_order, kernel_size=3, groups=groups, padding=0,
                           stride=1, dilation=1, grid_size=grid_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Calcular la dimensión de entrada para la primera capa totalmente conectada
        self.flatten_dim = self.calculate_flatten_dim(input_size)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flatten_dim, num_outputs)

    def calculate_flatten_dim(self, num_features):
        # Crear un tensor dummy para pasar por la red y obtener el tamaño de la salida
        x = torch.zeros(1, 1, num_features)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class ConvKAN(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1,
            grid_size: int = 5,
            num_outputs: int = 15,
            ):
        # num_outputs: size of the output layer (15 unless set by the caller).
        # num_classes is accepted for compatibility and does not change the model.
        super(ConvKAN, self).__init__()

        self.conv1 = KANConv1DLayer(input_dim=input_channels, output_dim=32, spline_order=spline_order, kernel_size=5, groups=1, padding=2, stride=1,
                           dilation=1, grid_size=grid_size)
        self.conv2 = KANConv1DLayer(input_dim=32, output_dim=32, spline_order=spline_order, kernel_size=5, groups=groups, padding=2,
                           stride=1, dilation=1, grid_size=grid_size)
        self.conv3 =KANConv1DLayer(input_dim=32,output_dim=64, spline_order=spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=1, dilation=1, grid_size=grid_size)
        self.conv4 =KANConv1DLayer(input_dim=64, output_dim=64, spline_order=spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=1, dilation=1, grid_size=grid_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calcular el tamaño de la salida después de las capas convolucionales
        conv_output_size = input_size // 4  # Dividir por 4 debido a dos capas maxpool

        self.fc1 = nn.Linear(64 * conv_output_size, 256) 
        self.fc2 = nn.Linear(256, num_outputs)

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout3 = nn.Dropout(p=0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x