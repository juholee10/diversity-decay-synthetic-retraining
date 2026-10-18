"""Iterative retraining on verifier-filtered samples.

Starting from a CVAE trained on 500 real digits, each generation trains a fresh
conditional discriminator against the current model, keeps the top 10% of the
model's samples by discriminator score, and trains the next model only on those.
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
ITERATIONS = 50

utils.seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = utils.make_run_dir("filtered")

train, test = data_helper.load_mnist()
by_digit = data_helper.indices_by_digit(train[1])
test_loader = DataLoader(TensorDataset(*test), batch_size=128)
evaluator = metrics.SampleEvaluator(train, test, device)
real_train = [t.to(device) for t in train]

idx = data_helper.balanced_subset(by_digit, INIT_SIZE)
model = train_helper.train_cvae(DataLoader(TensorDataset(train[0][idx], train[1][idx]), batch_size=128, shuffle=True), device)

rows = []
for it in range(1, ITERATIONS + 1):
    start = time.time()
    disc = train_helper.train_discriminator(model, real_train, device)

    images, labels = data_helper.generate_filtered(model, disc, TRAIN_SIZE, KEEP_FRACTION)
    filtered = evaluator(images, labels)
    utils.save_preview_grid(images, labels, run_dir / "samples" / f"iter{it}.png")

    model = train_helper.train_cvae(DataLoader(TensorDataset(images, labels), batch_size=128), device)
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
