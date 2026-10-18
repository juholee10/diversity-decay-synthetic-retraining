import torch
from torch.utils.data import DataLoader

import data_helper
import models


def run_epoch(model, loader, device, optimizer=None):
    """One pass over the loader, training if an optimizer is given.

    Returns the per-sample loss and the per-sample averages of whatever
    model.loss reports.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, totals = 0.0, {}
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.view(-1, 784).to(device), y.to(device)
            loss, stats = model.loss(x, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in totals.items()}


def train_model(model, train_loader, device, epochs=200, lr=1e-3, patience=5):
    """Adam with early stopping once the training loss stops improving."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, stale = float("inf"), 0
    for _ in range(epochs):
        train_loss, _ = run_epoch(model, train_loader, device, optimizer)
        if train_loss < best:
            best, stale = train_loss, 0
        else:
            stale += 1
            if stale >= patience:
                break


def train_cvae(loader, device):
    model = models.CVAE().to(device)
    train_model(model, loader, device)
    return model


def train_discriminator(generator, real_train, device):
    """Fresh discriminator against the current generator: the real training set
    vs. as many generated samples."""
    train_loader = DataLoader(data_helper.discriminator_dataset(*real_train, generator), batch_size=128, shuffle=True)
    disc = models.ConditionalDiscriminator(dropout=0.1, label_smoothing=0.05)
    train_model(disc, train_loader, device)
    return disc
