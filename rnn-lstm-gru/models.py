import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, device, isbidirectional):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.isbidirectional = isbidirectional
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=self.isbidirectional)
        if self.isbidirectional:
            self.num_layers = self.num_layers * 2
            self.fc_out = nn.Linear(hidden_size * 2, 128)
        else:
            self.fc_out = nn.Linear(hidden_size, 128)
        self.fc_out1 = nn.Linear(128, 5)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)

        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded, (h0, c0))
        prediction = self.fc_out(outputs[-1, :, :])
        prediction = self.fc_out1(prediction)
        return prediction


class GRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, device, isbidirectional):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.isbidirectional = isbidirectional
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=self.isbidirectional)
        if self.isbidirectional:
            self.num_layers = self.num_layers * 2
            self.fc_out = nn.Linear(hidden_size * 2, 128)
        else:
            self.fc_out = nn.Linear(hidden_size, 128)
        self.fc_out1 = nn.Linear(128, 5)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)

        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded, h0)
        prediction = self.fc_out(outputs[-1, :, :])
        prediction = self.fc_out1(prediction)

        return prediction


class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, 5)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)

        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded, h0)
        prediction = self.fc_out(outputs[-1, :, :])

        return prediction
