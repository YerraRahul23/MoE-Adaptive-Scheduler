"""Proposed adaptive scheduler.

Will combine the RL agent (``rl_agent.py``) and the quantum-inspired
optimizer (``optimizer.py``) into a single adaptive scheduling framework:
the RL agent learns workload patterns and proposes dispatch decisions,
while the optimizer refines expert-to-GPU placement, adapting to changing
load conditions.
"""
