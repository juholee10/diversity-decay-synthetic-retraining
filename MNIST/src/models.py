import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),   # 28 -> 14
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),  # 14 -> 7
        )
        self.fc = nn.Linear(64 * 7 * 7 + num_classes, 2 * latent_dim)

    def forward(self, x, y):
        h = self.conv(x.view(-1, 1, 28, 28)).flatten(1)
        mu, logvar = self.fc(torch.cat([h, y], dim=1)).chunk(2, dim=1)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),              # 14 -> 28
        )

    def forward(self, z, y):
        h = self.fc(torch.cat([z, y], dim=1)).view(-1, 64, 7, 7)
        return self.deconv(h).flatten(1)  # logits


class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = ConvEncoder(latent_dim, num_classes)
        self.decoder = ConvDecoder(latent_dim, num_classes)

    def loss(self, x, y):
        """Negative ELBO summed over the batch (Bernoulli likelihood)."""
        y = F.one_hot(y, self.num_classes).float()
        mu, logvar = self.encoder(x, y)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        logits = self.decoder(z, y)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
        return recon + kld, {"recon": recon.item(), "kld": kld.item()}

    @torch.no_grad()
    def sample(self, digit, n):
        """n binary samples of the given digit, shape (n, 784)."""
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        y = F.one_hot(torch.full((n,), digit, device=device), self.num_classes).float()
        return torch.bernoulli(torch.sigmoid(self.decoder(z, y)))


class ConditionalDiscriminator(nn.Module):
    """Real vs. generated classifier conditioned on the digit.

    Targets are 11-dim: the one-hot digit followed by a real (1) / fake (0) flag.
    """

    def __init__(self, num_classes=10, dropout=0.0, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        dims = [784 + num_classes, 512, 256, 128, 64]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
            if dropout > 0 and i < 2:
                layers.append(nn.Dropout(dropout))
        layers.append(spectral_norm(nn.Linear(dims[-1], 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y_digit):
        return self.net(torch.cat([x.view(len(x), -1), y_digit], dim=1))

    def score(self, x, y_digit):
        return torch.sigmoid(self(x, y_digit))

    def loss(self, x, y):
        # CVAE samples are binary, so binarize everything or the discriminator
        # can just tell real from fake by the pixel values.
        x = (x > 0.5).float()
        y_digit, target = y[:, :self.num_classes], y[:, self.num_classes:]
        logits = self(x, y_digit)
        smoothed = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed, reduction="sum"), {}


class SimpleAutoencoder(nn.Module):
    """Embedding network for the sample-quality metrics (see metrics.py)."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

