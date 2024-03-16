import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_units, decoder_units):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, encoder_units[0])
        self.fc2 = nn.Linear(encoder_units[0], encoder_units[1])
        # self.fc3_mean = nn.Linear(encoder_units[1], latent_dim)
        self.fc3_mean = nn.Linear(encoder_units[1], latent_dim)
        self.fc3_logvar = nn.Linear(encoder_units[1], latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, decoder_units[0])
        self.fc5 = nn.Linear(decoder_units[0], decoder_units[1])
        self.fc6 = nn.Linear(decoder_units[1], input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        # return F.sigmoid(self.fc3_mean(h2)), F.hardtanh(self.fc3_logvar(h2), min_val=-4.5, max_val=0.)
        return self.fc3_mean(h2), self.fc3_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)
        # return torch.relu(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
