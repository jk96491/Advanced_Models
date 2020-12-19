import torch
import torch.nn as nn
import numpy as np
from Models.VAE_Model.Encoder import Encoder
from Models.VAE_Model.Decoder import Decoder
from torch import optim


class vae(nn.Module):
    def __init__(self, args, device):
        super(vae, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.hidden_dim2 = args.hidden_dim2
        self.output_dim = args.output_dim
        self.latent_dim = args.latent_dim
        self.channels = args.channels

        self.encoder = Encoder(self.input_dim * self.channels, self.hidden_dim, self.hidden_dim2, self.output_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim2, self.hidden_dim, self.input_dim * self.channels)

        self.enc_mu = nn.Linear(self.output_dim, self.latent_dim)
        self.enc_log_sigma = nn.Linear(self.output_dim, self.latent_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        self.mse_loss = nn.MSELoss()

        self.z_mean = None
        self.z_sigma = None

        self.device = device
        self.to(self.device)

    def sampling_latent(self, encoder_output):
        mu = self.enc_mu(encoder_output)
        log_sigma = self.enc_log_sigma(encoder_output)

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z.to(self.device)

    def forward(self, inputs):
        encoder_output = self.encoder(inputs)
        latent = self.sampling_latent(encoder_output)
        return self.decoder(latent)

    def learn(self, inputs):
        decoder_output = self.forward(inputs)
        latent_loss = self.get_latent_loss(self.z_mean, self.z_sigma)
        loss = self.mse_loss(decoder_output, inputs) + latent_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_latent_loss(self, z_mean, z_std):
        mean_sq = z_mean * z_mean
        stdd_sq = z_std * z_std
        return 0.5 * torch.mean(mean_sq + stdd_sq - torch.log(stdd_sq) - 1)








