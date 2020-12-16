import torch
import torch.nn as nn
import numpy as np
from VAE_Model.Encoder import Encoder
from VAE_Model.Decoder import Decoder
from torch.autograd import Variable
from torch import optim


class vae(nn.Module):
    def __init__(self, args):
        super(vae, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.hidden_dim2 = args.hidden_dim2
        self.output_dim = args.output_dim
        self.latent_dim = args.latent_dim

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.hidden_dim2, self.output_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim2, self.hidden_dim, self.input_dim)

        self.enc_mu = nn.Linear(self.output_dim, self.latent_dim)
        self.enc_log_sigma = nn.Linear(self.output_dim, self.latent_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        self.mse_loss = nn.MSELoss()

    def sampling_latent(self, hidden):
        mu = self.enc_mu(hidden)
        log_sigma = self.enc_log_sigma(hidden)

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        latent = mu + sigma * Variable(std_z, requires_grad=False)

        return latent

    def forward(self, state):
        hidden = self.encoder(state)
        latent = self.sampling_latent(hidden)
        return self.decoder(latent)

    def learn(self, inputs):
        dec = self.forward(inputs)
        latent_loss = self.get_latent_loss(self.z_mean, self.z_sigma)
        loss = self.mse_loss(dec, inputs) + latent_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)








