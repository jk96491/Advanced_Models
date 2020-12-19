import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, out_dim):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim2),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
