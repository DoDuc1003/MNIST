import torch
import torch.nn as nn
def prepare_model(model):
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    return loss_func, optim

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out