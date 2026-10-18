"""Sample-quality metrics for generated digits.

FID and coverage are computed in the bottleneck of an autoencoder trained on
real MNIST instead of Inception features, which are a poor fit for 28x28
grayscale digits. The Vendi score uses the same embedding.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset

import models


@torch.no_grad()
def embed(encoder, images, batch_size=256):
    return torch.cat([encoder(x) for x in images.split(batch_size)])


def gaussian_stats(features):
    mu = features.mean(dim=0)
    centered = features - mu
    return mu, centered.T @ centered / (len(features) - 1)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm((sigma1 @ sigma2).cpu().numpy(), disp=False)[0].real
    covmean = torch.from_numpy(covmean).to(sigma1)
    return (diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)).item()


def knn_radii(features, k):
    """Distance from each point to its k-th nearest neighbour, excluding itself."""
    return torch.cdist(features, features).kthvalue(k + 1, dim=1).values


def coverage(real, fake, real_radii):
    """Coverage (Naeem et al., 2020): fraction of real samples with a generated
    sample inside their k-NN ball."""
    dist = torch.cdist(real, fake)
    return (dist.min(dim=1).values < real_radii).float().mean().item()


def vendi_score(features):
    """Effective number of distinct samples (Friedman & Dieng, 2023), using cosine similarity."""
    x = F.normalize(features, dim=1)
    eigvals = torch.linalg.eigvalsh(x @ x.T / len(x)).clamp(min=0)
    p = eigvals[eigvals > 1e-12]
    p = p / p.sum()
    return torch.exp(-(p * p.log()).sum()).item()


def train_autoencoder(images, device, embedding_dim, epochs=30):
    model = models.SimpleAutoencoder(embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(images), batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        total = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss = F.binary_cross_entropy(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
    print(f"autoencoder reconstruction loss: {total / len(images):.4f}")
    return model.eval()


class SampleEvaluator:
    """Compares generated digits with the real MNIST test set."""

    def __init__(self, train, test, device, embedding_dim=64, k=10):
        self.device = device
        self.encoder = train_autoencoder(train[0], device, embedding_dim).encoder
        self.real = embed(self.encoder, test[0].to(device))
        self.real_stats = gaussian_stats(self.real)
        self.real_radii = knn_radii(self.real, k)

    @torch.no_grad()
    def __call__(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        fake = embed(self.encoder, images)
        vendi = np.mean([vendi_score(fake[labels == c]) for c in range(10)])
        return {
            "fid": frechet_distance(*self.real_stats, *gaussian_stats(fake)),
            "coverage": coverage(self.real, fake, self.real_radii),
            "vendi": float(vendi),
        }
