from torch import nn
import torch
import sys
import torch.nn.functional as F

# directory reach
# sys.path.append('../kan_convolutional')

class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_outputs=15):
        # num_outputs: size of the output layer (15 unless set by the caller).
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(5, 5, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Calcular la dimensión de entrada para la primera capa totalmente conectada
        self.flatten_dim = self.calculate_flatten_dim(num_features)
        
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
    
class SimpleCNN_2(nn.Module):
    def __init__(self, num_features, num_outputs=15):
        # num_outputs: size of the output layer (15 unless set by the caller).
        super(SimpleCNN_2, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(5, 25, kernel_size=3, padding=0)

        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Calcular la dimensión de entrada para la primera capa totalmente conectada
        self.flatten_dim = self.calculate_flatten_dim(num_features)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flatten_dim, 245)
        self.fc2 = nn.Linear(245, num_outputs)

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
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleLinear(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SimpleLinear, self).__init__()
        # Definir una sola capa lineal con 'input_features' entradas y 10 salidas
        self.linear = nn.Linear(input_features, 10)
        self.flatten = nn.Flatten()


    def forward(self, x):
        # Pasar los datos a través de la capa lineal
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x
