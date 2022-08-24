import torch
import torch.nn as nn

import Config

features = Config.features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # decoder 
        self.decoder =  nn.Sequential(
            nn.Linear(in_features=features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=784),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.decoder(z)