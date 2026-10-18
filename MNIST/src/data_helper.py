import random

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets


def load_mnist(root="./data"):
    """(images, labels) for the train and test splits. Images are (N, 1, 28, 28) in [0, 1]."""
    splits = []
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        splits.append((ds.data.unsqueeze(1).float().div(255), ds.targets))
    return splits


def indices_by_digit(labels):
    """Shuffled dataset indices for each digit."""
    by_digit = {d: [] for d in range(10)}
    for i, y in enumerate(labels.tolist()):
        by_digit[y].append(i)
    for d in range(10):
        random.shuffle(by_digit[d])
    return by_digit


def balanced_subset(by_digit, size):
    return [i for d in range(10) for i in by_digit[d][:size // 10]]


def sample_real(images, labels, by_digit, n):
    """Fresh draw of n real images, n // 10 per digit."""
    idx = [i for d in range(10) for i in random.sample(by_digit[d], n // 10)]
    return images[idx], labels[idx]


@torch.no_grad()
def generate_balanced(model, n):
    """n samples from the CVAE, n // 10 per digit, ordered by digit."""
    model.eval()
    per_digit = n // 10
    images = torch.cat([model.sample(d, per_digit) for d in range(10)]).view(-1, 1, 28, 28)
    labels = torch.arange(10, device=images.device).repeat_interleave(per_digit)
    return images, labels


@torch.no_grad()
def generate_filtered(model, discriminator, n, keep_fraction, pool_size=6000):
    """n samples (n // 10 per digit) that pass the discriminator, in random order.

    Samples are drawn pool_size at a time per digit, and only the top
    keep_fraction of each draw by discriminator score is kept.
    """
    model.eval()
    discriminator.eval()
    per_digit = n // 10
    images = []
    for digit in range(10):
        kept, count = [], 0
        while count < per_digit:
            x = model.sample(digit, pool_size)
            y = F.one_hot(torch.full((pool_size,), digit, device=x.device), 10).float()
            scores = discriminator.score(x, y).squeeze(1)
            x = x[scores >= torch.quantile(scores, 1 - keep_fraction)][:per_digit - count]
            kept.append(x)
            count += len(x)
        images.append(torch.cat(kept))
    images = torch.cat(images).view(-1, 1, 28, 28)
    labels = torch.arange(10, device=images.device).repeat_interleave(per_digit)
    perm = torch.randperm(len(images))
    return images[perm], labels[perm]


def discriminator_dataset(real_images, real_labels, generator):
    """Real images against the same number of generated ones, with 11-dim targets
    (one-hot digit + real flag) as expected by ConditionalDiscriminator."""
    fake_images, fake_labels = generate_balanced(generator, len(real_images))
    images = torch.cat([real_images, fake_images])
    digits = F.one_hot(torch.cat([real_labels, fake_labels]), 10).float()
    is_real = torch.zeros(len(images), 1, device=images.device)
    is_real[:len(real_images)] = 1
    return TensorDataset(images, torch.cat([digits, is_real], dim=1))
