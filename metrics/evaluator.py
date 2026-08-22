"""Metrics evaluator.

Will calculate and aggregate evaluation metrics from simulation runs:

- TTFT: Time To First Token
- TPOT: Time Per Output Token
- Throughput: completed requests / tokens per second
- GPU utilization: compute usage across the cluster
- Queue waiting time: time requests spend waiting before execution
- Expert load balance: distribution of expert workload across GPUs
"""
