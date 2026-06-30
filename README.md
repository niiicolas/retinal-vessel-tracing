# Policy-Based Skeleton Tracing for Retinal Blood Vessels

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/pragyy/datascience-readme-template)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Bachelor Thesis** — *Reinforcement Learning for Retinal Vessel Skeletonization: A Policy-Driven Approach*
> **Authors:** Nicolas Fankhauser & Ravidu Nakandalage 
> **Institution:** ZHAW Wädenswil, Institute of Computational Life Sciences

Accurate extraction of retinal vessel centrelines is essential for quantifying vascular morphology associated with systemic diseases like diabetic retinopathy and hypertension. Traditional methods and deep segmentation networks typically produce centrelines via per-pixel classification followed by morphological thinning, offering no explicit connectivity guarantee. 

This repository implements a **Reinforcement Learning (RL) framework** that treats centreline extraction as a sequential Markov decision process (MDP). By deploying a PPO-trained actor-critic policy that moves step-by-step along each vessel, this approach constructs connected, topologically faithful skeletons *by design*, outperforming classical and supervised baselines under severe pathological domain shifts.

---

## Overview

### Pipeline at a glance

![Pipeline](pipeline.png)

### Key Features & Methodology

* **Tracing-as-Policy:** An RL agent (CNN encoder) outputs discrete world-frame steps. The resulting trajectory inherently forms a continuous skeleton, bridging minor contrast drops without relying on heavy post-processing heuristics.
* **Attention U-Net Seed Detector:** Provides spatially informed starting points for the RL agent, filtering out background noise and predicting vessel radius and orientation.
* **21-Channel Observation:** The agent receives a $65\times65$ local patch containing RGB crops, U-Net centreline priors, distance-transform geometry, tracing history and a multi-scale wide context crop.
* **Multi-Objective Reward Function:** The agent is guided by a dense reward signal prioritizing directed progress along unvisited vessels, with penalties for going off-track or revisiting covered pixels.
* **Curriculum & Imitation Learning:** The policy is warm-started using behaviour cloning (imitation learning) on expert traces, followed by PPO training governed by a continuous curriculum that gradually introduces complex, thin capillaries.

---

## 📂 Code Structure

```bash
├── config.py                # Single source of truth for hyperparameters (MODEL_CONFIG)
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

## Evaluation Metrics
 
The performance of the tracing pipelines is evaluated on geometric accuracy and topological consistency:
- **F1@τ / Precision / Recall** (at τ ∈ {1, 2, 3} px): Measures centreline overlap within a tolerance band. **F1@2 px** is the primary geometric metric.
- **clDice (Centreline Dice):** Balances tracking accuracy with network capture; the primary metric for topological connectivity and vessel overlap.
- **Betti-0 Error:** Measures the absolute difference in the number of connected components between the prediction and the ground truth. Reported both before (`raw`) and after (`post`) gap-closing to quantify reliance on post-processing.
- **HD95:** 95th-percentile Hausdorff distance, evaluating the maximum path deviation.
- **IoU:** Region overlap (Note: IoU naturally favours thick, area-filling baselines and is less reflective of one-pixel-wide skeleton quality).

---

## Results

The proposed RL agent is benchmarked against a classical Frangi filter pipeline and a supervised Centreline U-Net. Across standard validation sets and held-out external test sets (DRIVE and DR-HAGIS), the policy-driven approach consistently achieves the highest spatial precision and topological integrity.

### Main Comparative Performance
*Best results per split are highlighted in bold.*

| Split / Method | F1@2px | Precision | Recall | clDice | HD95 | Betti-0 (raw → post) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation (n=144)** | | | | | | |
| Frangi Filter | 0.449 | 0.628 | 0.366 | 0.386 | 79.5 | 20.5 → 5.2 |
| Centreline U-Net | 0.702 | 0.667 | **0.745** | 0.751 | 28.3 | 34.8 → 21.6 |
| **RL Agent** | **0.740** | **0.885** | 0.642 | **0.777** | **22.1** | **20.3 → 9.7** |
| **DRIVE (n=20)** | | | | | | |
| Frangi Filter | 0.492 | 0.384 | 0.701 | 0.373 | 31.8 | 5.2 → 4.5 |
| Centreline U-Net | 0.681 | 0.620 | **0.765** | 0.738 | 32.8 | 36.9 → 15.8 |
| **RL Agent** | **0.696** | **0.749** | 0.656 | **0.742** | **25.5** | **24.1 → 9.6** |
| **DR-HAGIS (n=40)** | | | | | | |
| Centreline U-Net | 0.571 | 0.473 | **0.732** | 0.374 | 50.5 | 19.0 → 9.7 |
| **RL Agent** | **0.615** | **0.607** | 0.631 | **0.389** | **40.6** | **18.1 → 7.6** |

> **Note on DR-HAGIS:** Under severe pathological domain shift, classical heuristic methods experience catastrophic tracking failure. The RL agent demonstrates superior structural robustness, reducing the Hausdorff distance (HD95) compared to standard pixel-wise classifiers.

---
 
## Acknowledgements

This work was supervised by **Dr. Norman Juchler** and **Fabio Muso** from the **Institute of Computational Life Sciences, ZHAW Wädenswil**.

We would also like to thank **Dr. Rui Santos** from the **Stadtspital Zürich (Augenklinik)** for his valuable clinical input and support throughout the project.


