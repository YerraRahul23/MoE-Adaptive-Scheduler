"""Simulation environment for MoE LLM serving.

Provides :class:`SimulationEnvironment` to orchestrate simulated request arrivals,
GPU queues, discrete-event request execution (prefill and decode phases), and
request completion lifecycle in a multi-GPU Mixture-of-Experts serving cluster.

All timestamps and durations represent **simulated time** in seconds. This module
does not execute CUDA operations or interface with physical GPU hardware.
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Set

from config.config import Config
from simulation.gpu import GPU
from simulation.request import InferenceRequest


@dataclass
class _ActiveExecution:
    """Internal tracker for a request actively executing on a GPU.

    Attributes:
        request: The active inference request.
        gpu_id: ID of the GPU executing the request.
        first_token_time: Exact simulated timestamp at which prefill finishes and
            the first token is generated.
        completion_time: Exact simulated timestamp at which generation completes.
    """

    request: InferenceRequest
    gpu_id: int
    first_token_time: float
    completion_time: float


class SimulationEnvironment:
    """Discrete-event simulation environment for MoE LLM serving.

    Manages cluster GPUs, queues, request lifecycle, and simulated time advancement.

    Request Lifecycle:
        1. **Pending**: Unarrived requests waiting until ``arrival_time <= current_time``.
        2. **Arrived**: Available for schedulers to inspect and dispatch.
        3. **Queued**: Assigned to a specific GPU queue (bounded by ``queue_capacity``).
        4. **Active**: Undergoing simulated prefill and token generation on a GPU.
        5. **Completed**: Processing finished with exact completion timestamps recorded.

    Queue Capacity Semantics:
        ``ClusterConfig.queue_capacity`` strictly bounds the number of **waiting requests**
        in a GPU's queue (``gpu.queue_length()``). It does not count requests currently
        in active execution slots.

    Simulated GPU Indicators:
        - **Utilization**: ``100.0 * (active_requests / max_active_per_gpu)`` clamped to [0.0, 100.0].
        - **KV Cache Usage**: Proportional to active token count clamped to [0.0, 100.0].
        - **Expert Workload**: Proportional to active requests routed to experts, clamped to [0.0, 1.0].
        All indicators are CPU-simulated estimates and do not represent hardware measurements.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        requests: Optional[List[InferenceRequest]] = None,
        prompt_processing_rate: float = 1000.0,
        token_generation_rate: float = 100.0,
        max_active_per_gpu: int = 1,
        max_kv_cache_tokens: int = 8192,
    ) -> None:
        """Initialize the simulation environment.

        Args:
            config: Master configuration. Defaults to ``Config()``.
            requests: Optional initial list of inference requests to load.
            prompt_processing_rate: Simulated prefill speed in prompt tokens per second.
                Must be a positive, finite number.
            token_generation_rate: Simulated decode speed in output tokens per second.
                Must be a positive, finite number.
            max_active_per_gpu: Maximum concurrent requests a GPU can process simultaneously.
                Must be a positive integer.
            max_kv_cache_tokens: Simulated KV cache token capacity per GPU for usage metrics.
                Must be a positive integer.

        Raises:
            ValueError: If configuration or parameters violate constraints.
        """
        if config is None:
            config = Config()
        elif not isinstance(config, Config):
            raise ValueError(f"config must be an instance of Config, got {type(config).__name__}")

        if (
            isinstance(prompt_processing_rate, bool)
            or not isinstance(prompt_processing_rate, (int, float))
            or not math.isfinite(prompt_processing_rate)
            or prompt_processing_rate <= 0.0
        ):
            raise ValueError(
                f"prompt_processing_rate must be a positive finite number, got {prompt_processing_rate!r}"
            )

        if (
            isinstance(token_generation_rate, bool)
            or not isinstance(token_generation_rate, (int, float))
            or not math.isfinite(token_generation_rate)
            or token_generation_rate <= 0.0
        ):
            raise ValueError(
                f"token_generation_rate must be a positive finite number, got {token_generation_rate!r}"
            )

        if (
            isinstance(max_active_per_gpu, bool)
            or not isinstance(max_active_per_gpu, int)
            or max_active_per_gpu <= 0
        ):
            raise ValueError(
                f"max_active_per_gpu must be a positive integer, got {max_active_per_gpu!r}"
            )

        if (
            isinstance(max_kv_cache_tokens, bool)
            or not isinstance(max_kv_cache_tokens, int)
            or max_kv_cache_tokens <= 0
        ):
            raise ValueError(
                f"max_kv_cache_tokens must be a positive integer, got {max_kv_cache_tokens!r}"
            )

        self.config: Config = config
        self.prompt_processing_rate: float = float(prompt_processing_rate)
        self.token_generation_rate: float = float(token_generation_rate)
        self.max_active_per_gpu: int = max_active_per_gpu
        self.max_kv_cache_tokens: int = max_kv_cache_tokens

        # Simulated GPU cluster
        self.gpus: Dict[int, GPU] = {
            gpu_id: GPU(gpu_id=gpu_id) for gpu_id in self.config.cluster.gpu_ids
        }

        # Simulated time and lifecycle
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.stop_reason: Optional[str] = None

        # Request state collections
        self._all_registered_ids: Set[int] = set()
        self.pending_requests: List[InferenceRequest] = []
        self.arrived_requests: List[InferenceRequest] = []
        self._active_executions: Dict[int, List[_ActiveExecution]] = {
            gpu_id: [] for gpu_id in self.gpus
        }
        self.completed_requests: List[InferenceRequest] = []

        if requests is not None:
            self.add_requests(requests)

    @property
    def total_requests(self) -> int:
        """Total number of registered requests across all states."""
        return len(self._all_registered_ids)

    def add_requests(self, requests: List[InferenceRequest]) -> None:
        """Register a collection of inference requests into the simulation.

        Requests are validated for unique IDs and sorted in non-decreasing
        order of arrival time. Requests whose arrival time is <= ``current_time``
        immediately transition to ``arrived_requests``.

        Args:
            requests: List of :class:`InferenceRequest` instances.

        Raises:
            ValueError: If any request is invalid, duplicate, or malformed.
        """
        if not isinstance(requests, list):
            raise ValueError(f"requests must be a list, got {type(requests).__name__}")

        for req in requests:
            if not isinstance(req, InferenceRequest):
                raise ValueError(
                    f"All items must be InferenceRequest instances, got {type(req).__name__}"
                )
            if req.request_id in self._all_registered_ids:
                raise ValueError(f"Duplicate request_id {req.request_id} detected")
            if req.is_completed():
                raise ValueError(f"Cannot register already completed request {req.request_id}")

            self._all_registered_ids.add(req.request_id)
            if req.arrival_time <= self.current_time:
                self.arrived_requests.append(req)
            else:
                self.pending_requests.append(req)

        # Ensure pending requests are sorted by (arrival_time, request_id)
        self.pending_requests.sort(key=lambda r: (r.arrival_time, r.request_id))

    def get_arrived_requests(self) -> List[InferenceRequest]:
        """Return a shallow copy of arrived, unscheduled requests."""
        return list(self.arrived_requests)

    def get_gpus(self) -> Dict[int, GPU]:
        """Return the dictionary of simulated GPUs mapped by GPU ID."""
        return self.gpus

    def assign_request(self, request: InferenceRequest, gpu_id: int) -> None:
        """Assign an arrived request to a specific GPU queue.

        Args:
            request: The arrived inference request to dispatch.
            gpu_id: ID of the target GPU.

        Raises:
            ValueError: If ``gpu_id`` is invalid, the request is not in
                ``arrived_requests``, or the target GPU's queue capacity is full.
        """
        if gpu_id not in self.gpus:
            available = sorted(list(self.gpus.keys()))
            raise ValueError(f"Invalid gpu_id {gpu_id}. Available GPUs: {available}")

        if request not in self.arrived_requests:
            raise ValueError(
                f"Request {request.request_id} is not available in arrived_requests "
                f"(assigned_gpu={request.assigned_gpu}, completed={request.is_completed()})"
            )

        gpu = self.gpus[gpu_id]
        capacity = self.config.cluster.queue_capacity
        if capacity is not None and gpu.queue_length() >= capacity:
            raise ValueError(
                f"GPU {gpu_id} queue capacity exceeded (limit: {capacity}, current: {gpu.queue_length()})"
            )

        self.arrived_requests.remove(request)
        request.assigned_gpu = gpu_id
        gpu.add_request(request)

    def is_finished(self) -> bool:
        """Check whether all requests have completed and the cluster is idle.

        Returns:
            ``True`` if pending, arrived, queued, and active requests are all empty.
        """
        if len(self.pending_requests) > 0 or len(self.arrived_requests) > 0:
            return False
        for gpu_id, gpu in self.gpus.items():
            if gpu.queue_length() > 0 or len(self._active_executions[gpu_id]) > 0:
                return False
        return True

    def _ingest_arrivals(self) -> None:
        """Move newly arrived requests from pending to arrived based on current time."""
        while self.pending_requests and self.pending_requests[0].arrival_time <= self.current_time:
            self.arrived_requests.append(self.pending_requests.pop(0))

    def _complete_active_executions(self) -> None:
        """Identify and finalize requests whose exact completion timestamp has passed."""
        for gpu_id, active_list in self._active_executions.items():
            still_active: List[_ActiveExecution] = []
            for exec_item in active_list:
                req = exec_item.request
                # Populate first_token_time once prefill has completed in simulated time
                if req.first_token_time is None and self.current_time >= exec_item.first_token_time:
                    req.first_token_time = exec_item.first_token_time

                if exec_item.completion_time <= self.current_time:
                    # Preserve exact calculated timestamps
                    req.completion_time = exec_item.completion_time
                    if req.first_token_time is None:
                        req.first_token_time = exec_item.first_token_time
                    self.completed_requests.append(req)
                else:
                    still_active.append(exec_item)
            self._active_executions[gpu_id] = still_active

    def _activate_queued_requests(self) -> None:
        """Pop waiting requests from GPU queues into available active execution slots."""
        for gpu_id, gpu in self.gpus.items():
            active_list = self._active_executions[gpu_id]
            while len(active_list) < self.max_active_per_gpu and gpu.queue_length() > 0:
                req = gpu.queue[0]
                gpu.remove_request(req)

                req.start_time = self.current_time
                req.queue_wait_time = max(0.0, req.start_time - req.arrival_time)

                prefill_duration = req.prompt_length / self.prompt_processing_rate
                first_token_time = req.start_time + prefill_duration

                decode_duration = req.output_length / self.token_generation_rate
                completion_time = first_token_time + decode_duration

                exec_item = _ActiveExecution(
                    request=req,
                    gpu_id=gpu_id,
                    first_token_time=first_token_time,
                    completion_time=completion_time,
                )
                active_list.append(exec_item)

    def _update_gpu_metrics(self) -> None:
        """Update simulated utilization, KV cache usage, and expert workload for all GPUs."""
        for gpu_id, gpu in self.gpus.items():
            active_list = self._active_executions[gpu_id]
            gpu.active_requests = len(active_list)

            if gpu.active_requests == 0:
                gpu.utilization = 0.0
                gpu.kv_cache_usage = 0.0
                gpu.expert_workload = 0.0
            else:
                # Simulated compute utilization in [0.0, 100.0]
                util = 100.0 * (gpu.active_requests / self.max_active_per_gpu)
                gpu.utilization = max(0.0, min(util, 100.0))

                # Simulated KV cache usage in [0.0, 100.0]
                active_tokens = sum(item.request.total_tokens() for item in active_list)
                cache_pct = (active_tokens / self.max_kv_cache_tokens) * 100.0
                gpu.kv_cache_usage = max(0.0, min(cache_pct, 100.0))

                # Simulated MoE expert workload in [0.0, 1.0]
                active_experts = gpu.active_requests * self.config.cluster.experts_per_token
                workload = active_experts / self.config.cluster.num_experts
                gpu.expert_workload = max(0.0, min(workload, 1.0))

    def step(
        self,
        time_delta: Optional[float] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        """Advance the simulation by a discrete time step.

        Execution Order:
            1. Ingest newly arrived requests (``arrival_time <= current_time``).
            2. Complete active requests whose ``completion_time <= current_time``.
            3. Invoke scheduler (if provided) to dispatch unscheduled arrived requests.
               If no scheduler is provided, arrived requests remain unscheduled.
            4. Dispatch queued requests to idle active GPU slots.
            5. Re-check for immediate completions (e.g. zero duration).
            6. Update simulated GPU metrics.
            7. Advance ``current_time`` by ``time_delta``.

        Args:
            time_delta: Simulated time step to advance in seconds. If ``None``,
                defaults to ``config.simulation.time_step``.
            scheduler: Optional scheduler callable or object with a ``schedule(env)``
                method.

        Raises:
            ValueError: If ``time_delta`` is not a positive finite number.
        """
        dt = self.config.simulation.time_step if time_delta is None else time_delta
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not math.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError(f"time_delta must be a positive finite number, got {dt!r}")
        dt = float(dt)

        # 1. Ingest arrivals
        self._ingest_arrivals()

        # 2. Complete previously active requests
        self._complete_active_executions()

        # 3. Invoke scheduler if provided
        if scheduler is not None:
            if callable(scheduler):
                scheduler(self)
            elif hasattr(scheduler, "schedule") and callable(scheduler.schedule):
                scheduler.schedule(self)
            else:
                raise ValueError("scheduler must be callable or expose a schedule(env) method")

        # 4. Dispatch queued requests to active slots
        self._activate_queued_requests()

        # 5. Check if any newly started requests complete within the same instant
        self._complete_active_executions()

        # 6. Update GPU metrics
        self._update_gpu_metrics()

        # 7. Advance time
        self.current_time += dt
        self.step_count += 1

    def run(
        self,
        scheduler: Optional[Any] = None,
        max_steps: int = 100_000,
    ) -> str:
        """Run the simulation until completion or a stopping condition is reached.

        Terminates safely on:
            - ``"completed"``: All requests have finished execution.
            - ``"max_duration_reached"``: Simulated time exceeds ``config.simulation.max_duration``.
            - ``"max_steps_reached"``: Step count reaches ``max_steps``.
            - ``"stalled"``: Cluster is idle, no future arrivals remain, but unscheduled
              arrived requests exist without progressing.

        All uncompleted requests remain preserved in their current state collections
        (``pending_requests``, ``arrived_requests``, ``gpu.queue``, or active executions).

        Args:
            scheduler: Optional scheduler callable or object with a ``schedule(env)``
                method.
            max_steps: Maximum step iterations before halting. Must be a positive integer.

        Returns:
            The termination reason string.

        Raises:
            ValueError: If ``max_steps`` is not a positive integer.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")

        if self.is_finished():
            self.stop_reason = "completed"
            return self.stop_reason

        while self.step_count < max_steps:
            if (
                self.config.simulation.max_duration is not None
                and self.current_time >= self.config.simulation.max_duration
            ):
                self.stop_reason = "max_duration_reached"
                return self.stop_reason

            prev_completed = len(self.completed_requests)
            prev_arrived = len(self.arrived_requests)

            self.step(scheduler=scheduler)

            if self.is_finished():
                self.stop_reason = "completed"
                return self.stop_reason

            # Progress check: stall detection
            has_active = any(len(a) > 0 for a in self._active_executions.values())
            has_pending = len(self.pending_requests) > 0
            has_queued = any(gpu.queue_length() > 0 for gpu in self.gpus.values())

            if not has_active and not has_pending and not has_queued and len(self.arrived_requests) > 0:
                if len(self.completed_requests) == prev_completed and len(self.arrived_requests) == prev_arrived:
                    self.stop_reason = "stalled"
                    return self.stop_reason

        self.stop_reason = "max_steps_reached"
        return self.stop_reason

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the simulation state and GPU cluster.

        Returns:
            Dictionary detailing current time, step count, termination reason,
            request counts across all lifecycle states, and per-GPU metrics.
        """
        queued_count = sum(gpu.queue_length() for gpu in self.gpus.values())
        active_count = sum(len(a) for a in self._active_executions.values())

        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "stop_reason": self.stop_reason,
            "total_requests": self.total_requests,
            "completed_count": len(self.completed_requests),
            "pending_count": len(self.pending_requests),
            "arrived_count": len(self.arrived_requests),
            "queued_count": queued_count,
            "active_count": active_count,
            "is_finished": self.is_finished(),
            "gpus": {
                gpu_id: {
                    "load_score": gpu.load_score(),
                    "queue_length": gpu.queue_length(),
                    "active_requests": gpu.active_requests,
                    "utilization": gpu.utilization,
                    "kv_cache_usage": gpu.kv_cache_usage,
                    "expert_workload": gpu.expert_workload,
                }
                for gpu_id, gpu in self.gpus.items()
            },
        }
