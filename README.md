# Modeling Diversity Decay in Synthetic Retraining

Code for the CVAE/MNIST experiments in **"Modeling Diversity Decay in Synthetic Retraining"** by Juho Lee.

**Status:** under review at the [Journal of Youth Impact](https://journaljyi.org).
**Paper:** [project page](https://juholee10.github.io/publication/2026-diversity-decay) · [PDF (LaTeX version)](https://juholee10.github.io/files/DiversityDecayTex.pdf) · [PDF (Word version)](https://juholee10.github.io/files/DiversityDecayWord.pdf)

## About the paper

When generative models are retrained on their own synthetic data, the outputs drift away from the real distribution over generations. This is known as model collapse. Most work on collapse measures fidelity (how realistic the samples are). This paper looks at **diversity** separately and asks whether it decays in a predictable way.

The paper studies this in two settings:

1. **A Gaussian model (theory and simulation).** Each generation refits a 1-D Gaussian to samples from the previous one. The estimated variance is a martingale, so its expected value stays constant, yet it still converges to zero almost surely (Doob's martingale convergence theorem plus the strong law of large numbers on log-variance). In simulation, a fixed-radius verifier also makes the expected variance decay. Adding a verifier together with 10% real data each generation leads to a stable but reduced level of variance instead.
2. **A conditional VAE on MNIST (this repository).** A CVAE is trained on 500 real digits and then retrained on its own samples for 50 generations, under three conditions: no verifier, a verifier that keeps the top 10% of samples, and the verifier plus 10% real data.

### Main findings from the CVAE experiments

| Setting | Vendi Score | Coverage | FID | Held-out reconstruction loss |
|---|---|---|---|---|
| No verifier | 22.0 → 30.4 (+38%) | 0.192 → 0.026 (−87%) | 16.7 → 62.6 | 95.4 → 167.3 |
| Verifier (top 10%) | 19.7 → 7.4 (−62%) | ~0.30 → 0.33 | no clear trend (~15–27) | 93.4 → 150.6 |
| Verifier + 10% real data | 18.6 → ~10.5, then flat (−41%) | ~0.29 → 0.51 | no clear trend (~6–12) | flat (~85–89) |

- **With a verifier, diversity falls steadily while the samples still look clean.** The Vendi Score drops by 62%, and the drop resembles the decay seen in the Gaussian model.
- **Mixing in real data stops the decline, but not at the original level.** Diversity settles at a lower plateau and reconstruction loss stays flat.
- **Common metrics can miss this.** In the filtered settings, Coverage goes up and FID shows no trend while the Vendi Score falls. Coverage and the Vendi Score move in opposite directions in all three settings. The paper argues that a diversity-specific metric (such as the Vendi Score) should be tracked alongside a fidelity measure tied to real data (such as reconstruction loss).

Limitations noted in the paper: a single random seed, a small starting set (500 images), one mixing fraction and one filtering fraction, and metrics computed with a custom MNIST encoder rather than InceptionV3, so FID values are not comparable with other work.

## Experiments

All three experiments start from a CVAE trained on 500 real digits (50 per class) and run for 50 generations. Each generation trains a new CVAE from scratch on 10,000 images built from the previous model's samples, without keeping any earlier data (the "replace" setting).

| Script | Paper setting | Training data for each generation |
|---|---|---|
| `MNIST/scripts/unfiltered_experiment.py` | No verifier | All 10,000 raw samples |
| `MNIST/scripts/ELBO_experiment.py` | Verifier | Top 10% of samples by discriminator score, 1,000 per digit |
| `MNIST/scripts/filtered_mixing_experiment.py` | Verifier + real-data mixing | 9,000 filtered samples + 1,000 real MNIST images, drawn fresh each generation |

The verifier is an MLP discriminator conditioned on the digit. It is retrained every generation to tell the 60k real MNIST training images apart from the same number of samples from the current model. It only filters samples; it never sends gradients to the generator.

The Gaussian simulation is not included in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Needs Python 3.9+. MNIST is downloaded to `./data` on the first run.

## Running

```bash
python MNIST/scripts/unfiltered_experiment.py
python MNIST/scripts/ELBO_experiment.py
python MNIST/scripts/filtered_mixing_experiment.py
```

Settings are the constants at the top of each script. Use a GPU; a full run takes several hours. All runs use seed 0.

Outputs go to `MNIST/runs/<experiment>_<timestamp>/`, or to `MyDrive/verified_synthetic_data/MNIST/runs/` on Colab. If you start a script on Colab with `!python`, mount Drive from a notebook cell first. Set `RUN_ID` to choose the folder name instead of the timestamp.

Each run folder contains:

- `results.csv`: one row per generation, rewritten after each generation
- `plots/`: one line plot per column of `results.csv`
- `samples/`: 5 samples per digit from each generation (the filtered set in the two verifier experiments)

## Metrics

Generated samples are compared against the 10k MNIST test images. Instead of Inception features, the embedding is the 64-dimensional bottleneck of a small autoencoder trained on real MNIST.

- `fid`: Fréchet distance in that embedding
- `coverage`: Coverage from Naeem et al. (2020) with k = 10
- `vendi`: Vendi Score (cosine similarity) within each digit, averaged over the 10 digits
- `val_recon`: reconstruction loss of the new CVAE on the held-out test set

Row *i* describes the samples that generation *i* was trained on, and the test-set reconstruction loss of generation *i* itself. In the two verifier experiments the sample metrics are computed on the filtered set only (suffix `_filtered`); in the mixing run this covers just the synthetic part.

## Code layout

```
MNIST/
  scripts/   the three experiments above
  src/
    models.py       CVAE, conditional discriminator, metric autoencoder
    data_helper.py  MNIST loading, balanced and filtered sampling
    train_helper.py training loops (Adam, early stopping)
    metrics.py      FID, Coverage, Vendi Score
    utils.py        seeding, run folders, sample grids and plots
```

## Acknowledgements

The filtered-retraining setup and CVAE architecture are adapted from Yi et al., [*Escaping Model Collapse via Synthetic Data Verification*](https://arxiv.org/abs/2510.16657). This work was done through the Seoul Research Program, mentored by Professor Haifeng Xu (University of Chicago), with guidance from TA Bin Yu.
