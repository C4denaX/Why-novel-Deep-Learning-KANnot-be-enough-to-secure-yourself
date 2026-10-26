from torch import nn
import torch
import torch.nn.functional as F
import sys
# directory reach
# sys.path.append('../kan_convolutional')

class ConvNet1D(nn.Module):
    def __init__(self, input_size, num_outputs=15):
        # num_outputs: size of the output layer (15 unless set by the caller).
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        # Calcular el tamaño de la salida después de las capas convolucionales
        conv_output_size = input_size // 4  # Dividir por 4 debido a dos capas maxpool

        self.fc1 = nn.Linear(64 * conv_output_size, 256) 
        self.fc2 = nn.Linear(256, num_outputs)

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