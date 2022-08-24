import torch
import torch.nn as nn

import Config

features = Config.features

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=features*2)
        )
    
    def forward(self, x):
        return self.encoder(x)