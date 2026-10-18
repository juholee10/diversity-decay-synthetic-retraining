"""Iterative retraining without a verifier.

Baseline for ELBO_experiment.py: every generation trains directly on 10,000 raw
samples from the previous model. No discriminator is involved, so the results
have a single set of metric columns instead of *_unfiltered / *_filtered.
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
ITERATIONS = 50

utils.seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = utils.make_run_dir("unfiltered")

train, test = data_helper.load_mnist()
by_digit = data_helper.indices_by_digit(train[1])
test_loader = DataLoader(TensorDataset(*test), batch_size=128)
evaluator = metrics.SampleEvaluator(train, test, device)

idx = data_helper.balanced_subset(by_digit, INIT_SIZE)
model = train_helper.train_cvae(DataLoader(TensorDataset(train[0][idx], train[1][idx]), batch_size=128, shuffle=True), device)

rows = []
for it in range(1, ITERATIONS + 1):
    start = time.time()
    images, labels = data_helper.generate_balanced(model, TRAIN_SIZE)
    scores = evaluator(images, labels)
    utils.save_preview_grid(images, labels, run_dir / "samples" / f"iter{it}.png")

    model = train_helper.train_cvae(DataLoader(TensorDataset(images, labels), batch_size=128, shuffle=True), device)
    _, stats = train_helper.run_epoch(model, test_loader, device)

    rows.append({
        "iteration": it,
        **scores,
        "val_recon": stats["recon"],
    })
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "results.csv", index=False)
    utils.save_metric_plots(results, run_dir / "plots")

    print(f"[{it}/{ITERATIONS}] FID {scores['fid']:.2f}, coverage {scores['coverage']:.3f}, "
          f"vendi {scores['vendi']:.2f}, test recon {stats['recon']:.2f} ({time.time() - start:.0f}s)")
