import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(name):
    """runs/<name>_<RUN_ID>/ under MNIST/, or on Google Drive when running in Colab.

    RUN_ID defaults to the current time; set the env var to choose it yourself.
    """
    try:
        from google.colab import drive
    except ImportError:
        base = Path(__file__).resolve().parents[1]
    else:
        # drive.mount() needs the notebook kernel. If the script is launched with
        # `!python ...`, mount Drive from a notebook cell first.
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")
        base = Path("/content/drive/MyDrive/verified_synthetic_data/MNIST")

    run_id = os.environ.get("RUN_ID") or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    run_dir = base / "runs" / f"{name}_{run_id}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    print(f"saving to {run_dir}")
    return run_dir


def save_preview_grid(images, labels, path, per_class=5):
    images, labels = images.cpu().view(-1, 28, 28), labels.cpu()
    fig, axes = plt.subplots(10, per_class, figsize=(1.5 * per_class, 15))
    for digit in range(10):
        idx = (labels == digit).nonzero().flatten()[:per_class]
        for j, ax in enumerate(axes[digit]):
            ax.axis("off")
            if j < len(idx):
                ax.imshow(images[idx[j]].numpy(), cmap="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_metric_plots(results, plot_dir):
    """One line plot per column of the results table, redrawn every iteration."""
    for column in results.columns.drop("iteration"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results["iteration"], results[column], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel(column)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{column}.png", dpi=150)
        plt.close(fig)
