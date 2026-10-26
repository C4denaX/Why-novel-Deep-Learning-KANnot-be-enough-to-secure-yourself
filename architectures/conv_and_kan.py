from torch import nn
import sys,torch
import torch.nn.functional as F

# sys.path.append('../kan_convolutional')

from kan_convolutional.KANLinear import KANLinear

class NormalConvsKAN(nn.Module):
    def __init__(self, num_features, grid_size=10, spline_order=3, num_outputs=15):
        # num_outputs: size of the output layer (15 unless set by the caller).
        super(NormalConvsKAN, self).__init__()
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Flatten layer
        self.flatten_dim = self.calculate_flatten_dim(num_features)
        
        self.flatten = nn.Flatten()

        # KAN layer
        self.kan1 = KANLinear(
            self.flatten_dim,
            num_outputs,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])

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
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)

        return x