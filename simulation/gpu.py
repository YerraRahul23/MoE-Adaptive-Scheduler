"""Simulated GPU model.

Defines the :class:`GPU` dataclass, representing the simulated state of a
single GPU in the MoE serving cluster. It tracks compute utilization,
KV cache usage, expert workload, waiting requests, and actively processed
requests, and exposes helper methods used by scheduling algorithms to
inspect load and manage their queues.

All values represent **simulated** state; there is no real GPU/CUDA
integration. ``load_score`` is only a simple baseline load representation
for the simulation, not the final heuristic or the proposed
quantum-inspired scheduler.
"""

from dataclasses import dataclass, field
from typing import List

from .request import InferenceRequest


@dataclass
class GPU:
    """A single simulated GPU serving MoE LLM inference requests.

    Attributes:
        gpu_id: Unique identifier for the simulated GPU. Must be >= 0.
        utilization: Current compute utilization as a percentage in
            [0.0, 100.0].
        kv_cache_usage: Current KV cache usage as a percentage in
            [0.0, 100.0].
        expert_workload: Current MoE expert workload on this GPU,
            normalized to [0.0, 1.0].
        queue: Inference requests waiting to be processed by this GPU.
        active_requests: Number of requests currently being processed.
            Must be >= 0.
    """

    # --- Core fields --------------------------------------------------------
    gpu_id: int
    utilization: float = 0.0
    kv_cache_usage: float = 0.0
    expert_workload: float = 0.0

    # --- Runtime/request tracking fields -------------------------------------
    queue: List[InferenceRequest] = field(default_factory=list)
    active_requests: int = 0

    def __post_init__(self) -> None:
        """Validate fields after construction.

        Raises:
            ValueError: If any field violates its constraints.
        """
        if self.gpu_id < 0:
            raise ValueError(
                f"gpu_id must be non-negative, got {self.gpu_id}"
            )
        if not 0.0 <= self.utilization <= 100.0:
            raise ValueError(
                f"utilization must be between 0.0 and 100.0, got "
                f"{self.utilization}"
            )
        if not 0.0 <= self.kv_cache_usage <= 100.0:
            raise ValueError(
                f"kv_cache_usage must be between 0.0 and 100.0, got "
                f"{self.kv_cache_usage}"
            )
        if not 0.0 <= self.expert_workload <= 1.0:
            raise ValueError(
                f"expert_workload must be between 0.0 and 1.0, got "
                f"{self.expert_workload}"
            )
        if self.active_requests < 0:
            raise ValueError(
                f"active_requests must be non-negative, got "
                f"{self.active_requests}"
            )

    def queue_length(self) -> int:
        """Return the current number of requests waiting in the queue.

        Returns:
            The number of requests in ``queue``.
        """
        return len(self.queue)

    def add_request(self, request: InferenceRequest) -> None:
        """Add an inference request to this GPU's waiting queue.

        Args:
            request: The request to append to the queue.
        """
        self.queue.append(request)

    def remove_request(self, request: InferenceRequest) -> None:
        """Remove a request from the queue.

        Args:
            request: The request to remove from the queue.

        Raises:
            ValueError: If the request is not present in the queue.
        """
        if request not in self.queue:
            raise ValueError(
                f"request {request.request_id} not found in queue of "
                f"GPU {self.gpu_id}"
            )
        self.queue.remove(request)

    def is_idle(self) -> bool:
        """Check whether the GPU is currently idle.

        Returns:
            ``True`` if no requests are being processed and the queue is
            empty, ``False`` otherwise.
        """
        return self.active_requests == 0 and len(self.queue) == 0

    def load_score(self) -> float:
        """Return a simple normalized load score in [0.0, 1.0].

        Baseline weighted combination of utilization, KV cache pressure,
        expert workload, and queue pressure:

            load_score = 0.40 * (utilization / 100)
                       + 0.25 * (kv_cache_usage / 100)
                       + 0.20 * expert_workload
                       + 0.15 * min(queue_length / 10, 1.0)

        This is only a baseline load representation for the simulation;
        it is not the final heuristic or the proposed quantum-inspired
        scheduler.

        Returns:
            The normalized load score, clamped to [0.0, 1.0].
        """
        queue_factor: float = min(self.queue_length() / 10.0, 1.0)
        score: float = (
            0.40 * (self.utilization / 100.0)
            + 0.25 * (self.kv_cache_usage / 100.0)
            + 0.20 * self.expert_workload
            + 0.15 * queue_factor
        )
        return max(0.0, min(score, 1.0))
