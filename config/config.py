"""Project-wide configuration for the MoE Adaptive Scheduler.

This module will centralize configuration used by the rest of the project:

- Hardware settings: number of simulated GPUs, GPU capacity, memory limits.
- MoE model settings: number of experts, experts per token/request.
- Simulation parameters: time-step resolution, queue limits, run duration.
- Workload settings: dataset paths, arrival rates, light/medium/heavy presets.
- Experiment configuration: baselines vs. proposed scheduler, seeds, repeats.
- Output settings: paths under ``results/`` and ``logs/``.
"""
