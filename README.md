# Policy-Based Skeleton Tracing for Retinal Blood Vessels

An RL-based approach to extract vessel centerlines from retinal fundus images.

## Overview

This project implements a reinforcement learning agent that traces blood vessel centerlines in retinal photographs. Instead of using traditional segmentation + post-processing, the agent learns to navigate through vessels, producing connected skeletons by construction.

### Key Features

- **RL-based tracing**: PPO agent with CNN encoder and optional LSTM for temporal context
- **Tolerance-aware rewards**: Smooth rewards based on distance to ground truth centerlines
- **Seed detection**: CNN-based detection of endpoints and junctions for trace initialization
- **Frontier-based coverage**: Strategy to cover all branches while maintaining connectivity
- **Curriculum learning**: Progressive difficulty increase during training
- **Comprehensive evaluation**: Multiple metrics including F1@τ, clDice, and topology measures

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vessel-tracing.git
cd vessel-tracing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
