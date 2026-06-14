# Policy-Based Skeleton Tracing for Retinal Blood Vessels

A reinforcement-learning agent that extracts vessel **centerlines** from retinal
fundus photographs by *tracing* them, rather than segmenting pixels and
post-processing. A PPO policy learns to walk along vessels, producing connected
skeletons by construction.

> **Final certified model (v12):** centerline **F1@2px = 0.670** on the
> validation split, **0.666** on held-out DRIVE and **0.618** on held-out
> DRHAGIS — all certified leak-free (see [Leak-free certification](#leak-free-certification)).
> Full experiment record in [`ABLATIONS.md`](ABLATIONS.md).

---

## Overview

Traditional pipelines segment vessels per-pixel and then thin/skeletonise, which
breaks connectivity at crossings and thin vessels. Here the agent instead
**navigates** the vessel tree: starting from detected seeds, it takes discrete
steps along the centerline, and the resulting trajectory *is* the skeleton — so
the output is connected by construction.

### Pipeline at a glance

```
                 ┌──────────────────────┐
 fundus image →  │  SeedDetector        │  Attention U-Net, multi-task:
                 │  (centerline U-Net)  │  centerline-prob + endpoint/junction seeds
                 └──────────┬───────────┘
                            │  centerline prior + seeds
                            ▼
 ┌────────────────────────────────────────────────────────────┐
 │  Imitation (BC)  →  PPO (RL)  →  FrontierTracer (inference)  │
 │  warm-start         600 iters     ring seeds + frontier      │
 │  the policy         curriculum     coverage, snap, gate      │
 └────────────────────────────────────────────────────────────┘
                            │
                            ▼
              connected centerline skeleton
              scored: F1@τ, clDice, Betti-0, gt_edge_cov80, HD95
```

### Key ideas

- **Tracing-as-policy** — a PPO agent (CNN encoder, optional LSTM) outputs
  discrete world-frame steps; the trajectory is the skeleton.
- **21-channel egocentric observation** — local RGB crop, U-Net centerline
  prior, distance-transform/tangent geometry, visited/coverage maps, topology
  memory, and a multi-scale wide crop.
- **Directed-progress reward** — the primary per-step term rewards moving toward
  *uncovered* vessel; coverage, frontier-extension, off-vessel and revisit terms
  shape behaviour; a terminal F-β term credits the trace's contribution.
- **Frontier-based inference** — ring seeds + a frontier coverage strategy fan
  out across all branches; **snap-to-centerline** fixes sub-pixel drift and a
  **vessel-gate** drops off-vessel points (the single biggest quality lever,
  ≈ +0.39 F1).
- **No GT at inference** — every reported number is *certified leak-free* by a
  corrupt-GT byte-identity test.

---

## Repository layout

```
config.py                  Single source of truth for all hyperparameters (MODEL_CONFIG)
data/
  dataloader.py            Combined multi-dataset loader; train/val split + held-out test
  centerline_extraction.py GT centerline / skeleton extraction (cached)
  fundus_preprocessor.py   FOV crop, resize-and-pad, normalisation
  DRIVE/ FIVES/ HRF/ ...    Raw datasets (images + manual masks)
environment/
  vessel_env.py            Gymnasium tracing environment (obs, step, termination)
  observation.py           ObservationBuilder — the 21-channel egocentric stack
  reward.py                Per-step + terminal reward terms
  frontier_tracer.py       Inference-time multi-seed frontier tracer (snap + gate)
  seeding_utils.py         Seed placement, ring seeds, FOV-scale handling
  vec_env.py               Vectorised env workers for PPO
models/
  policy_network.py        ActorCriticNetwork (CNN/LSTM encoder + policy/value heads)
  seed_detector.py         Multi-task Attention U-Net (centerline + endpoints/junctions)
  unet.py / unet_blocks.py U-Net building blocks
  frangi.py                Frangi vesselness baseline
  greedy_tracer.py         Greedy steepest-ascent tracer baseline
training/
  imitation.py             Behaviour-cloning warm start
  ppo.py                   PPO trainer (GAE, curriculum, entropy anneal)
  curriculum.py            easy → medium → full difficulty stages
  seed_detector_trainer.py Seed-detector training loop
scripts/                   Entry points (see below)
evaluation/
  metrics.py               F1@τ, clDice, Betti-0, gt_edge_cov80, HD95
  scoring.py               Shared scorer so RL & baselines are metric-comparable
weights/<run>/             Checkpoints + logs, namespaced per run (RVT_RUN_NAME)
results/<run>/             Per-image metrics, summaries, visualisations
ABLATIONS.md               Full experiment / ablation record (read for any number)
*.sh                       SLURM batch scripts (train_v12.sh is the final recipe)
```

---

## Installation

The project targets Python 3.9 on a SLURM cluster (RHEL8, single GPU).

```bash
module load gcc/9.4.0-pe5.34 python/3.9.12-pe5.34   # cluster modules
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

A conda alternative is provided in [`environment.yml`](environment.yml), and the
cluster venv build is automated in [`venv.sh`](venv.sh)
(`sbatch venv.sh`).

Core dependencies: PyTorch ≥ 2.0, Gymnasium, scikit-image, `skan` (skeleton
analysis), OpenCV, NetworkX, NumPy/SciPy, pandas, matplotlib (see
[`requirements.txt`](requirements.txt)).

---

## Data

Seven public fundus datasets live under [`data/`](data/). The loader
([`data/dataloader.py`](data/dataloader.py)) combines five into a balanced
train/val pool and holds two out entirely as external test sets:

| role | datasets |
|---|---|
| **train / val** | FIVES, STARE, CHASEDB1, HRF, LES-AV |
| **test (held-out)** | DRIVE, DRHAGIS |

The loader applies FOV cropping, aspect-preserving resize-and-pad, disk-caches
GT centerlines and U-Net priors, and (for training) uses a
`WeightedRandomSampler` so each dataset contributes equally despite size
imbalance. Train/val is a deterministic split of the sorted samples.

---

## Usage

All hyperparameters live in [`config.py`](config.py) (`MODEL_CONFIG`). **The repo
defaults reproduce the final v12 model** — no environment variables or flags are
needed. (`RVT_RUN_NAME` only namespaces `weights/<run>/` and `results/<run>/` so
parallel jobs don't clobber each other.)

Run scripts as modules from the repo root with the venv active.

### 1. Train the seed detector (perception)

```bash
python -m scripts.train_seed_detector
```

Trains the multi-task Attention U-Net. Its **centerline-prob head is reused as
the RL centerline prior**, so the same checkpoint (`weights/<run>/seed_detector.pt`)
serves both seeding and the observation prior. This must exist before PPO.

### 2. Train the policy (imitation → PPO)

```bash
python -m scripts.train_imitation   # behaviour-cloning warm start
python -m scripts.train_ppo         # PPO, 600 iterations, curriculum
```

### 3. Evaluate / test the traced skeletons

```bash
python -m scripts.run_rl_tracing --eval            # validation split
python -m scripts.run_rl_tracing --test            # held-out DRIVE + DRHAGIS
python -m scripts.run_rl_tracing --eval --corrupt-gt   # leak-free certification
```

Outputs land in `results/<run>/RL_tracing_e2e/<split>/` (per-image
`metrics_e2e.csv`, a summary, and trajectory visualisations).

### End-to-end on the cluster

[`train_v12.sh`](train_v12.sh) runs the whole recipe as one SLURM job —
imitation → PPO 600 → eval → GT-ablation certification — and prints a
PASS/FAIL certification verdict:

```bash
sbatch train_v12.sh
```

---

## Baselines

All baselines run at the RL agent's settings and are scored through the **shared
scorer** ([`evaluation/scoring.py`](evaluation/scoring.py)), so their numbers are
directly comparable to the RL agent:

```bash
python -m scripts.run_frangi        --eval   # Frangi vesselness + centerline
python -m scripts.run_greedytracer  --eval   # greedy steepest-ascent tracer
python -m scripts.run_cnn           --eval   # centerline U-Net (segment + thin)
```

---

## Evaluation metrics

Reported per image and aggregated (see [`config.py`](config.py) `METRIC_COLS` and
[`evaluation/metrics.py`](evaluation/metrics.py)):

- **F1@τ / precision / recall** at τ ∈ {1, 2, 3} px — centerline overlap at a
  tolerance band (headline = **F1@2px**).
- **clDice** — centerline-aware Dice.
- **Betti-0 error** — connected-component (topology) error, raw and post-processed.
- **gt_edge_cov80** — fraction of GT edges ≥ 80 % covered (recall of structure).
- **HD95** — 95th-percentile Hausdorff distance.

---

## Leak-free certification

Earlier high scores were inflated by **ground-truth leakage at inference**
(off-track termination, bridge keep-tests, gap reseeding, coverage normalisation
all peeked at GT). Removing the four leaks cost ≈ 0.07 F1 but makes the numbers
trustworthy.

Certification is a **corrupt-GT byte-identity test**: feed the tracer garbage GT
(`--corrupt-gt`); if the metrics are byte-for-byte identical to the normal run,
the prediction provably does not depend on GT. Every recordable run has a
`*_gtcorrupt` twin and a PASS verdict. **Only certified, leak-free numbers are
reportable.**

---

## Results

Final model **v12** (certified, leak-free; config = repo default):

| split | F1@2px | P@2 | R@2 | clDice | gt_edge_cov80 | Betti-0 | cert |
|---|---|---|---|---|---|---|---|
| **val** | **0.670** | 0.840 | 0.563 | 0.487 | 0.581 | 13.0 | PASS |
| **test — DRIVE** | **0.666** | 0.778 | 0.589 | 0.483 | 0.662 | 14.8 | PASS |
| **test — DRHAGIS** | **0.618** | 0.674 | 0.572 | 0.339 | 0.645 | 4.6 | PASS |

Test ≈ val ⇒ the policy generalises to datasets never seen in training.

### Headline findings (full detail in [`ABLATIONS.md`](ABLATIONS.md))

1. **World-frame + step-2 actions** unblocked imitation (BC acc 0.60 → 0.73) and
   were the precondition for any end-to-end learning.
2. The inference **vessel-gate is the largest single lever (≈ +0.39 F1)** — an
   off-vessel false-positive suppressor, not a model change; snap-to-centerline
   adds ≈ +0.04.
3. **Removing GT leakage cost ≈ 0.07 F1** but makes the result honest; certified
   v12 (0.670) recovers nearly the best-ever *leaky* score (0.690).
4. **Cheap episode termination hurts recall** (the v11 regression): the agent
   over-traces only when stopping early is costly — the termination penalties are
   load-bearing.
5. Recall plateaus at ≈ 0.56 because the result is **perception-rich but
   trace-poor**: the centerline perception lights up ≈ 95 % of GT, but only
   ≈ 56 % is traced. The remaining gap is tracer exploration / seeding, not
   perception or the τ gate — which bounds what reward tuning can buy and points
   future work at seeding and connectivity.

---

## Configuration notes

- **Single source of truth.** Edit knobs directly in `MODEL_CONFIG` in
  [`config.py`](config.py); env-var sweep overrides were removed. The only runtime
  toggles are `--corrupt-gt` (certification) and `RVT_RUN_NAME` (output namespacing).
- **Observation stack:** 21 channels — toggled via the `environment.use_*` flags
  (multi-scale crop, topology memory, U-Net prior, etc.). Changing the channel
  layout invalidates existing checkpoints.
- **On-vessel signal:** dense U-Net vesselness with τ = 0.30 drives off-track
  termination and reward gating (leak-free; the `gt` signal is debug-only).
- **Workers:** PPO uses `n_envs = 8` — each worker holds a CUDA context for the
  prior U-Net, so 16 OOMs a 64 GB GPU.

---

## Citation / context

Developed as a research project (ZHAW) on policy-based skeleton tracing for
retinal vasculature. For the complete version history, negative results, and
ablations, see [`ABLATIONS.md`](ABLATIONS.md) and the `RETRAIN_PLAN_v*.md` notes.
</content>
</invoke>
