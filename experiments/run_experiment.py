"""Experiment runner.

Will orchestrate end-to-end experiments that compare the baseline schedulers
(Round Robin, Least Loaded, Heuristic) against the proposed Adaptive
Scheduler over the configured workloads, persisting raw outputs to
``results/raw/``, summaries to ``results/reports/``, and graphs to
``results/plots/``.
"""
