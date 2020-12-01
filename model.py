import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Actor, self).__init__()
        self.hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.Tanh(),
        )

        self.p = nn.Linear(self.hidden, out_channels)
        self.v = nn.Linear(self.hidden, 1)

    def forward(self, x):
        out = self.fc(x)

        p = self.p(out)
        v = self.v(out)

        return p, v


class Critic(nn.Module):
    def __init__(self, in_channels, args):
        super(Critic, self).__init__()
        self.hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.Tanh(),
        )

        self.v = nn.Linear(self.hidden, 1)

    def forward(self, x):
        out = self.fc(x)
        out = self.v(out)

        return out
