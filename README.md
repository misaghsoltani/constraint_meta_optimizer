# Constraint-Aware Meta-Optimizer for Policy Gradient in Reinforcement Learning

This repository contains a framework for meta-optimization of constrained policy gradient methods in reinforcement learning. The framework supports safety constraints, and bilevel meta-learning. The approach is based on meta-learning update rules for both policy and dual variables (Lagrange multipliers) in constrained Markov decision processes (CMDPs), with evaluation on Safety-Gymnasium benchmark tasks.

## Table of Contents

- [Constraint-Aware Meta-Optimizer for Policy Gradient in Reinforcement Learning](#constraint-aware-meta-optimizer-for-policy-gradient-in-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [How It Works](#how-it-works)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Using pixi (Recommended)](#using-pixi-recommended)
    - [Using pip/conda](#using-pipconda)
  - [Usage](#usage)
  - [Citation](#citation)
  - [License](#license)

---

## Overview

Many reinforcement learning (RL) applications require agents to maximize expected return while satisfying explicit safety constraints. This project implements a meta-optimizer that learns how to update both policy and constraint parameters in such settings. The optimizer is trained using bilevel optimization, where the outer loop meta-learns optimizer parameters and the inner loop adapts policy and constraint variables. The framework supports both learnable (LSTM-based) and structured (PID) meta-optimizers.

## How It Works

- **Problem Setting:** The agent operates in a CMDP, maximizing expected reward while keeping expected cost (safety violations) below a threshold.
- **Meta-Optimization:** The meta-optimizer learns update rules for policy and dual variables, parameterized either as an LSTM or as a PID controller.
- **Bilevel Training:** The outer loop updates meta-parameters to maximize a penalized return objective, based on the final policy after several inner updates.
- **Evaluation:** The approach is evaluated on Safety-Gymnasium tasks, with metrics including reward, cost, constraint satisfaction, and learning stability.
<!-- - **Baselines:** The framework includes implementations of PPO-Lagrangian (primal-dual), PID Lagrangian, and CPO (Constrained Policy Optimization) for comparison. -->

## Project Structure

```bash
constraint_meta_optimizer/
├── meta_trainer.py      # Main entry for meta-training and evaluation
├── meta_optimizer.py    # LSTM-based and PID meta-optimizer implementations
├── agent.py             # Primal-dual agent logic (policy, constraints, updates)
├── policy.py            # Policy network architectures
├── envs.py              # Environment wrappers and utilities
├── logger.py            # Logging, TensorBoard, CSV, and plotting utilities
config/
├── default.yaml         # Default experiment configuration
outputs/, runs/          # Experiment results, logs, and checkpoints
```

## Installation

### Prerequisites

 The project requires Python 3.10 and PyTorch 2.0+. See [pyproject.toml](pyproject.toml) for dependencies.

### Using [pixi](https://pixi.sh/latest/) (Recommended)

[Pixi](https://pixi.sh/latest/) is a package manager for Python that simplifies the installation of dependencies and project management. First, [install `pixi`](https://pixi.sh/latest/#installation). To install the project using pixi, follow these steps:

```bash
git clone https://github.com/misaghsoltani/constraint_meta_optimizer.git
cd constraint_meta_optimizer
pixi install
pixi shell
```

### Using pip/conda

```bash
conda create -n cmo python=3.10
conda activate cmo
pip install --upgrade pip
pip install \
  "torch>=2.0" \
  "safety-gymnasium @ git+https://github.com/PKU-MARL/Safety-Gymnasium.git" \
  "gymnasium>=0.28.1,<0.29" \
  "numpy>=1.23.5,<2" \
  "tqdm>=4.67.1,<5" \
  "pyyaml>=6.0.2,<7" \
  "hydra-core>=1.3" \
  "omegaconf>=2.3.0,<3" \
  "matplotlib>=3.10.1,<4" \
  "seaborn>=0.13.2,<0.14" \
  "pandas>=2.2.3,<3" \
  "tensorboard>=2.19.0,<3"
pip install -e .
```

## Usage

Experiment settings are controlled via YAML config files in `config/`. See `config/default.yaml` for an example. Key parameters include:

- `env_ids`: List of environment IDs for meta-training
- `total_meta_iters`, `inner_rollouts`, `meta_batch_size`: Meta-training parameters
- `hidden_layers`, `hidden_size`: Policy network architecture
- `alpha_theta`, `alpha_lambda`: Optimizer hyperparameters
- `init_KP`, `init_KI`, `init_KD`, `init_etaR`, `init_etaC`, `init_lambda`: PID Lagrangian parameters
- `meta_lr`, `unroll_steps`, `gamma`, `cost_limit`, `mu`: Meta-optimizer training
- `lstm_layers`: LSTM meta-optimizer parameters
- `log_interval`, `checkpoint_interval`: Logging and checkpointing

Parameters can be overridden from the command line, e.g.:

```bash
# if installed using pixi
cmo meta_lr=5e-4 device=cpu

# or

python -m constraint_meta_optimizer.meta_trainer meta_lr=5e-4 device=cpu
```

Results, logs, and checkpoints are saved in the `runs/` and `outputs/` directories.

To launch TensorBoard:

```bash
tensorboard --logdir runs/
```

## Citation

```bibtex
@misc{soltani2025constraintmetaopt,
  author  = {Misagh Soltani},
  title   = {Constraint-Aware Meta-Optimizer for Policy Gradient},
  year    = {2025},
  url     = {https://github.com/misaghsoltani/constraint_meta_optimizer}
}
```

## License

See [LICENSE](LICENSE) for details.
