"""Simulation environment.

Will manage the overall simulation loop for MoE LLM serving: maintaining the
request queue, the pool of simulated GPUs, dispatching requests through a
chosen scheduler, stepping time forward, and recording events for evaluation.
"""
