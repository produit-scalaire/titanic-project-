import torch
import torch.nn as nn

class neural_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x