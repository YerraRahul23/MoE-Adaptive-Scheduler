"""Workload generator for LLM inference requests.

Will later generate or load synthetic and benchmark LLM inference workloads,
turning the JSON datasets in ``workload/datasets/`` into ordered streams of
inference requests with configurable arrival patterns and intensity levels
(light, medium, heavy).
"""
