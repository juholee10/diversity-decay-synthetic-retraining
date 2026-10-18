"""Verifier-filtered retraining with real data mixed in.

Same loop as ELBO_experiment.py, but each generation's 10,000 training images
are 90% filtered samples and 10% real MNIST images, drawn fresh every generation.
The *_filtered metrics only look at the synthetic part, so they stay comparable
with the run without mixing.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_helper
import metrics
import train_helper
import utils

INIT_SIZE = 500
TRAIN_SIZE = 10_000
KEEP_FRACTION = 0.1
REAL_FRACTION = 0.1
ITERATIONS = 50

n_real = round(TRAIN_SIZE * REAL_FRACTION)
n_synthetic = TRAIN_SIZE - n_real

utils.seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = utils.make_run_dir("filtered_mixing")

train, test = data_helper.load_mnist()
by_digit = data_helper.indices_by_digit(train[1])
test_loader = DataLoader(TensorDataset(*test), batch_size=128)
evaluator = metrics.SampleEvaluator(train, test, device)
real_train = [t.to(device) for t in train]

idx = data_helper.balanced_subset(by_digit, INIT_SIZE)
model = train_helper.train_cvae(DataLoader(TensorDataset(train[0][idx], train[1][idx]), batch_size=128, shuffle=True), device)
print(f"training set per generation: {n_synthetic} filtered + {n_real} real")

rows = []
for it in range(1, ITERATIONS + 1):
    start = time.time()
    disc = train_helper.train_discriminator(model, real_train, device)

    images, labels = data_helper.generate_filtered(model, disc, n_synthetic, KEEP_FRACTION)
    filtered = evaluator(images, labels)
    utils.save_preview_grid(images, labels, run_dir / "samples" / f"iter{it}.png")

    # Binarize the real images so they look like the CVAE's Bernoulli samples.
    real_images, real_labels = data_helper.sample_real(*train, by_digit, n_real)
    images = torch.cat([(real_images > 0.5).float().to(device), images])
    labels = torch.cat([real_labels.to(device), labels])
    perm = torch.randperm(len(images))

    model = train_helper.train_cvae(DataLoader(TensorDataset(images[perm], labels[perm]), batch_size=128), device)
    _, stats = train_helper.run_epoch(model, test_loader, device)

    rows.append({
        "iteration": it,
        **{f"{k}_filtered": v for k, v in filtered.items()},
        "val_recon": stats["recon"],
    })
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "results.csv", index=False)
    utils.save_metric_plots(results, run_dir / "plots")

    print(f"[{it}/{ITERATIONS}] FID {filtered['fid']:.2f}, coverage {filtered['coverage']:.3f}, "
          f"vendi {filtered['vendi']:.2f}, test recon {stats['recon']:.2f} ({time.time() - start:.0f}s)")
