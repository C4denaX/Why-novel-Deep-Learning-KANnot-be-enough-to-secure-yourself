import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNMalware_Model1(nn.Module):
    def __init__(self, input_dim, embedding_dim=100, hidden_dim=100, output_dim=20,
                 batch_size=256, num_layers=1, bidirectional=False, dropout=0):
        super(RNNMalware_Model1, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.fc_hidden_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          nonlinearity='relu', bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.fc_hidden_dim, self.output_dim)

    def forward(self, x):
        # (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        output, hidden = self.rnn(x)
        return self.fc(output[-1])



class LSTMMalware_Model1(nn.Module):
    def __init__(self, input_dim, embedding_dim=100, hidden_dim=100, output_dim=20,
                 batch_size=256, num_layers=1, bidirectional=False, dropout=0):
        super(LSTMMalware_Model1, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.fc_hidden_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.fc_hidden_dim, self.output_dim)

    def forward(self, x):
        # (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        output, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        fc_in = self.dropout_layer(hidden)
        return self.fc(fc_in)


class GRUMalware_Model1(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=20,
                 num_layers=1, bidirectional=False, dropout=0):
        super(GRUMalware_Model1, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.fc_hidden_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.fc_hidden_dim, self.output_dim)

    def forward(self, x):
        # (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        output, hidden = self.gru(x)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        fc_in = self.dropout_layer(hidden)
        return self.fc(fc_in)
    
class StackedMalware_Model1(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=20, num_layers=1, bidirectional=False, dropout=0, LG=True):
        super(StackedMalware_Model1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.fc_hidden_dim = self.hidden_dim * 2 if bidirectional else self.hidden_dim
        self.num_of_direction = 2 if bidirectional else 1
        self.LG = LG
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)

        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.fc_hidden_dim, self.output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
        output = x
        if self.LG:
            output, (hidden, cell) = self.lstm(output)
            output, hidden = self.gru(output)
        else:
            output, hidden = self.gru(output)
            output, (hidden, cell) = self.lstm(output)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        fc_in = self.dropout_layer(hidden)
        return self.fc(fc_in)