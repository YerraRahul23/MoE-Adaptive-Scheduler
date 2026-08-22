# moe-adaptive-scheduler

**Quantum-Inspired Adaptive Scheduling Framework for Efficient Mixture-of-Experts Large Language Model Serving**

## Project Description

This project is a CLI-based research and simulation framework for studying
scheduling in **Mixture-of-Experts (MoE)** large language model serving
systems. It will simulate clusters of GPUs serving LLM inference requests and
compare baseline scheduling policies against a proposed adaptive scheduler
that combines:

- A **Reinforcement Learning (RL) agent** that learns workload patterns and
  scheduling policies.
- A **Quantum-Inspired / QUBO-based optimizer** that refines expert-to-GPU
  placement under changing load conditions.

The goal is to reduce latency (TTFT, TPOT), improve throughput, balance expert
load across GPUs, and adapt to dynamic workloads better than static baselines.

> **Note:** This is not a web application. Everything runs from the command line.

## Proposed Architecture

```
Workload Generator ──► Simulation Environment ──► Scheduler ──► Simulated GPUs
                              │                                    │
                              └──────────► Metrics Evaluator ◄─────┘
                                               │
                                  Experiments / Results / Plots
```

1. **Workload layer** generates or loads light/medium/heavy LLM inference
   request streams from JSON datasets.
2. **Simulation layer** models requests, GPU state (utilization, queue length,
   KV cache usage, per-expert workload), and the discrete-event execution flow.
3. **Scheduler layer** decides where each request is dispatched:
   - Baselines: Round Robin, Least Loaded, Heuristic.
   - Proposed: RL agent + quantum-inspired optimizer (adaptive scheduler).
4. **Metrics layer** evaluates TTFT, TPOT, throughput, GPU utilization, queue
   waiting time, and expert load balance.
5. **Experiment layer** runs comparisons across schedulers/workloads and writes
   raw results, summary reports, and plots.

## Folder Structure

```
moe-adaptive-scheduler/
├── README.md
├── requirements.txt
├── main.py                     # Main CLI entry point
├── config/
│   └── config.py               # GPUs, simulation, workload, experiment settings
├── workload/
│   ├── generator.py            # Synthetic/benchmark workload generation
│   └── datasets/               # light.json, medium.json, heavy.json
├── simulation/
│   ├── request.py              # Inference request model
│   ├── gpu.py                  # Simulated GPU state
│   └── environment.py          # Overall simulation environment
├── schedulers/
│   ├── base_scheduler.py       # Common scheduler interface
│   ├── round_robin.py          # Baseline: Round Robin
│   ├── least_loaded.py         # Baseline: Least Loaded
│   ├── heuristic.py            # Baseline: heuristic policy
│   └── proposed/
│       ├── rl_agent.py         # RL component
│       ├── optimizer.py        # Quantum-inspired / QUBO optimizer
│       └── adaptive_scheduler.py  # Proposed combined scheduler
├── metrics/
│   └── evaluator.py            # TTFT, TPOT, throughput, utilization, etc.
├── experiments/
│   └── run_experiment.py       # Baseline vs. proposed comparisons
├── results/
│   ├── raw/                    # CSV/JSON experiment outputs
│   ├── reports/                # Summary reports
│   └── plots/                  # Generated graphs
├── tests/                      # Unit and integration tests
└── logs/                       # Runtime logs
```

## Current Development Status

**Scaffolding phase.** The repository currently contains only the project
structure and placeholder modules with docstrings describing their intended
responsibilities. No scheduling algorithms, RL agent, quantum-inspired
optimizer, simulation logic, or MoE inference have been implemented yet.

## Future Setup Instructions

Once implementation begins:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (to be populated in requirements.txt)
pip install -r requirements.txt

# 3. Run the CLI (entry point under construction)
python main.py --help

# 4. Run experiments
python -m experiments.run_experiment

# 5. Run tests
python -m unittest discover tests
```

The project can be opened directly in VS Code or Cursor; no additional editor
configuration is required.
