import torch
import torch.nn as nn

import Config
from Encoder import Encoder
from Decoder import Decoder

features = Config.features

# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var, mode):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """

        if mode == "train":

            std = torch.exp(0.5*log_var) # standard deviation
            eps = torch.randn_like(std) # `randn_like` as we need the same size
            sample = mu + (eps * std) # sampling as if coming from the input space
            
            return sample

        return mu
 
    def forward(self, x, mode="train"):

        # encoding
        x = self.encoder(x).view(-1, 2, features)
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var, mode=mode)
 
        # decoding
        x_hat = self.decoder(z)

        return x_hat, mu, log_var