import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.gelu(self.fc1(state))
        flat2 = F.gelu(self.fc2(flat1))
        flat3 = F.gelu(self.fc3(flat2))
        flat4 = F.gelu(self.fc4(flat3))
        actions = self.fc5(flat4)

        return actions
