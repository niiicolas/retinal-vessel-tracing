# Policy-Based Skeleton Tracing for Retinal Blood Vessels

> Bachelor Thesis — *Reinforcement Learning for Retinal Vessel Skeletonization: A Policy-Driven Approach*
> Nicolas Fankhauser & Ravidu Nakandalage · ZHAW Wädenswil, Institute of Computational Life Sciences

This repository implements a reinforcement learning framework that traces retinal vessel **centrelines** as a sequential decision process rather than via per-pixel classification. An attention U-Net seed detector proposes starting points; a PPO-trained actor–critic policy, warm-started with imitation learning, then walks pixel-by-pixel along each vessel, producing a **connected skeleton by construction** — no ground truth required at inference. The agent is benchmarked against a Frangi vesselness filter, a greedy heuristic tracer and a supervised centreline U-Net across five training datasets and two held-out external test sets (DRIVE, DR-HAGIS).

---

## Overview

### Pipeline at a glance

![Pipeline](pipeline.png)

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

## Repository Structure

```text
├── config.py                # Single source of truth for all hyperparameters (MODEL_CONFIG)
├── data/
│   ├── dataloader.py        # Combined multi-dataset loader; train/val split + held-out test
│   ├── centerline_ext.py    # GT centerline / skeleton extraction (cached)
│   └── fundus_prep.py       # FOV crop, resize-and-pad, normalisation
├── environment/
│   ├── vessel_env.py        # Gymnasium tracing environment (obs, step, termination)
│   ├── observation.py       # ObservationBuilder — the 21-channel egocentric stack
│   ├── reward.py            # Per-step + terminal reward terms
│   └── frontier_tracer.py   # Inference-time multi-seed frontier tracer (snap + gate)
├── models/
│   ├── policy_network.py    # ActorCriticNetwork (CNN encoder + policy/value heads)
│   ├── seed_detector.py     # Multi-task Attention U-Net
│   ├── frangi.py            # Frangi vesselness baseline
│   └── greedy_tracer.py     # Greedy steepest-ascent tracer baseline
├── training/
│   ├── imitation.py         # Behaviour-cloning warm start
│   ├── ppo.py               # PPO trainer (GAE, curriculum, entropy anneal)
│   └── curriculum.py        # Dynamic difficulty progression stages
├── scripts/                 # Execution entry points (see Usage)
└── evaluation/
    ├── metrics.py           # F1@τ, clDice, Betti-0, HD95 computations
    └── scoring.py           # Shared scorer so RL & baselines are metric-comparable

```

## Data

Training and validation:

| Dataset |
|-|
| FIVES |
| STARE |
| CHASE_DB1 |
| HRF |
| LES-AV |

External evaluation:

| Dataset | Purpose |
|-|-|
| DRIVE | Standard benchmark |
| DR-HAGIS | Pathological domain shift |

The external datasets are never used during training or hyperparameter tuning.

---

## Installation
 
```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
 
---
 
## Usage
 
All hyperparameters live in [`config.py`](config.py) (`MODEL_CONFIG`). Run scripts as modules from the repo root with the virtual environment active.
 
### 1. Train the seed detector
 
```bash
python -m scripts.train_seed_detector
```
 
### 2. Train the policy (imitation → PPO)
 
```bash
python -m scripts.train_imitation   # behaviour-cloning warm start
python -m scripts.train_ppo         # PPO, curriculum
```
 
### 3. Evaluate the traced skeletons
 
```bash
python -m scripts.run_rl_tracing --eval   # validation split
python -m scripts.run_rl_tracing --test   # held-out datasets (DRIVE, DR-HAGIS)
```
 
Outputs land in `results/<run>/RL_tracing_e2e/<split>/`: per-image `metrics_e2e.csv`, a summary table, and trajectory visualisations.
 
### Baselines
 
```bash
python -m scripts.run_frangi        --eval   # Frangi vesselness + centreline
python -m scripts.run_greedytracer  --eval   # greedy tracer
python -m scripts.run_cnn           --eval   # supervised centreline U-Net
```

---

## Evaluation metrics
 
- **F1@τ / precision / recall** at τ ∈ {1, 2, 3} px — centreline overlap within a tolerance band (headline metric: F1@2 px).
- **clDice** — centreline-aware Dice; the primary metric for vessel connectivity.
- **Betti-0 error** — connected-component (topology) error, reported raw and post gap-closing.
- **HD95** — 95th-percentile Hausdorff distance.
- **IoU** — region overlap (naturally favours thick, area-filling baselines over a one-pixel-wide skeleton).

---

## Results

Final model:

| split | F1@2px | P@2 | R@2 | clDice | gt_edge_cov80 | Betti-0 | cert |
|---|---|---|---|---|---|---|---|
| **val** | **0.670** | 0.840 | 0.563 | 0.487 | 0.581 | 13.0 | PASS |
| **test — DRIVE** | **0.666** | 0.778 | 0.589 | 0.483 | 0.662 | 14.8 | PASS |
| **test — DRHAGIS** | **0.618** | 0.674 | 0.572 | 0.339 | 0.645 | 4.6 | PASS |


---
 
## Acknowledgements

This work was supervised by **Dr. Norman Juchler** and **Fabio Muso** from the **Institute of Computational Life Sciences, ZHAW Wädenswil**.

We would also like to thank **Dr. Rui Santos** from the **Stadtspital Zürich (Augenklinik)** for his valuable clinical input and support throughout the project.

---
 
*This document was created with assistance from AI tools. Content has been reviewed and edited by the project authors.*

