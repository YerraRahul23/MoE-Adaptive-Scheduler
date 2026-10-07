"""Project-wide configuration for the MoE Adaptive Scheduler.

Provides structured, dataclass-based configuration for simulating Mixture-of-Experts
(MoE) large language model serving clusters:

- :class:`ClusterConfig`: Simulated GPU counts, expert capacity, and queues.
- :class:`WorkloadProfileConfig`: Presets for light, medium, and heavy workloads.
- :class:`WorkloadConfig`: Workload profiles and synthetic generator parameters.
- :class:`SimulationConfig`: Simulation step size, stopping conditions, and output paths.
- :class:`Config`: Top-level aggregated configuration container.

All values represent **simulated** environments and synthetic request streams.
This framework conducts discrete-event simulation and does not interface with
physical GPUs or measure actual CUDA performance.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ClusterConfig:
    """Configuration for the simulated GPU cluster and MoE model layout.

    Attributes:
        num_gpus: Number of simulated GPUs in the cluster. Must be >= 1.
        num_experts: Total number of MoE experts distributed across the cluster.
            Must be >= 1.
        experts_per_token: Number of active experts routed per token/request
            (Top-K routing). Must satisfy ``1 <= experts_per_token <= num_experts``.
        queue_capacity: Maximum number of waiting requests allowed in a GPU queue.
            If ``None``, queues are unbounded. If specified, must be >= 1.
    """

    num_gpus: int = 4
    num_experts: int = 8
    experts_per_token: int = 2
    queue_capacity: Optional[int] = 64

    def __post_init__(self) -> None:
        """Validate cluster parameters.

        Raises:
            ValueError: If any cluster configuration constraint is violated.
        """
        if isinstance(self.num_gpus, bool) or not isinstance(self.num_gpus, int) or self.num_gpus <= 0:
            raise ValueError(f"num_gpus must be a positive integer, got {self.num_gpus!r}")
        if isinstance(self.num_experts, bool) or not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError(f"num_experts must be a positive integer, got {self.num_experts!r}")
        if (
            isinstance(self.experts_per_token, bool)
            or not isinstance(self.experts_per_token, int)
            or not (1 <= self.experts_per_token <= self.num_experts)
        ):
            raise ValueError(
                f"experts_per_token must be an integer between 1 and num_experts ({self.num_experts}), "
                f"got {self.experts_per_token!r}"
            )
        if self.queue_capacity is not None:
            if (
                isinstance(self.queue_capacity, bool)
                or not isinstance(self.queue_capacity, int)
                or self.queue_capacity <= 0
            ):
                raise ValueError(
                    f"queue_capacity must be None or a positive integer, got {self.queue_capacity!r}"
                )

    @property
    def gpu_ids(self) -> List[int]:
        """Derived list of simulated GPU IDs [0, ..., num_gpus - 1]."""
        return list(range(self.num_gpus))


def _default_priority_weights() -> Dict[str, float]:
    """Default relative frequency weights for request priorities."""
    return {"low": 0.2, "normal": 0.6, "high": 0.2}


@dataclass
class WorkloadProfileConfig:
    """Configuration for a specific synthetic or benchmark workload profile.

    Attributes:
        name: Identifier for the profile (e.g. 'light', 'medium', 'heavy').
        num_requests: Total number of requests in the workload. Must be >= 1.
        arrival_interval_range: (min, max) interval between request arrivals
            in simulated seconds. Must satisfy ``0 <= min <= max``.
        prompt_length_range: (min, max) prompt tokens per request.
            Must satisfy ``1 <= min <= max``.
        output_length_range: (min, max) generation tokens per request.
            Must satisfy ``1 <= min <= max``.
        priority_weights: Mapping of priority levels ('low', 'normal', 'high')
            to non-negative sampling weights. Sum of weights must be > 0.
    """

    name: str
    num_requests: int
    arrival_interval_range: Tuple[float, float]
    prompt_length_range: Tuple[int, int]
    output_length_range: Tuple[int, int]
    priority_weights: Dict[str, float] = field(default_factory=_default_priority_weights)

    def __post_init__(self) -> None:
        """Validate workload profile parameters.

        Raises:
            ValueError: If any profile parameter violates defined constraints.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if isinstance(self.num_requests, bool) or not isinstance(self.num_requests, int) or self.num_requests <= 0:
            raise ValueError(f"num_requests must be a positive integer, got {self.num_requests!r}")

        # Validate arrival_interval_range
        if not isinstance(self.arrival_interval_range, (tuple, list)) or len(self.arrival_interval_range) != 2:
            raise ValueError(
                f"arrival_interval_range must be a 2-tuple of numbers (min, max), "
                f"got {self.arrival_interval_range!r}"
            )
        min_arr, max_arr = self.arrival_interval_range
        for val, label in ((min_arr, "minimum"), (max_arr, "maximum")):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"arrival_interval_range {label} must be a finite number, got {val!r}")
        if min_arr < 0.0:
            raise ValueError(f"arrival_interval_range minimum must be >= 0, got {min_arr}")
        if min_arr > max_arr:
            raise ValueError(
                f"arrival_interval_range minimum ({min_arr}) cannot exceed maximum ({max_arr})"
            )

        # Validate prompt_length_range
        if not isinstance(self.prompt_length_range, (tuple, list)) or len(self.prompt_length_range) != 2:
            raise ValueError(
                f"prompt_length_range must be a 2-tuple of integers (min, max), "
                f"got {self.prompt_length_range!r}"
            )
        min_prompt, max_prompt = self.prompt_length_range
        for val, label in ((min_prompt, "minimum"), (max_prompt, "maximum")):
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"prompt_length_range {label} must be an integer, got {val!r}")
        if min_prompt < 1:
            raise ValueError(f"prompt_length_range minimum must be >= 1 token, got {min_prompt}")
        if min_prompt > max_prompt:
            raise ValueError(
                f"prompt_length_range minimum ({min_prompt}) cannot exceed maximum ({max_prompt})"
            )

        # Validate output_length_range
        if not isinstance(self.output_length_range, (tuple, list)) or len(self.output_length_range) != 2:
            raise ValueError(
                f"output_length_range must be a 2-tuple of integers (min, max), "
                f"got {self.output_length_range!r}"
            )
        min_output, max_output = self.output_length_range
        for val, label in ((min_output, "minimum"), (max_output, "maximum")):
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"output_length_range {label} must be an integer, got {val!r}")
        if min_output < 1:
            raise ValueError(f"output_length_range minimum must be >= 1 token, got {min_output}")
        if min_output > max_output:
            raise ValueError(
                f"output_length_range minimum ({min_output}) cannot exceed maximum ({max_output})"
            )

        # Validate priority_weights
        if not isinstance(self.priority_weights, dict):
            raise ValueError(f"priority_weights must be a dictionary, got {type(self.priority_weights).__name__}")
        expected_keys = {"low", "normal", "high"}
        actual_keys = set(self.priority_weights.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"priority_weights must contain exactly keys {expected_keys}, got {actual_keys}"
            )
        total_weight = 0.0
        for k, w in self.priority_weights.items():
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
                raise ValueError(f"priority weight for '{k}' must be a finite number, got {w!r}")
            if w < 0.0:
                raise ValueError(f"priority weight for '{k}' must be non-negative, got {w}")
            total_weight += w
        if total_weight <= 0.0:
            raise ValueError(
                f"sum of priority_weights must be strictly positive, got {total_weight}"
            )


def _default_workload_profiles() -> Dict[str, WorkloadProfileConfig]:
    """Instantiate fresh default workload profiles for light, medium, and heavy workloads."""
    return {
        "light": WorkloadProfileConfig(
            name="light",
            num_requests=100,
            arrival_interval_range=(0.02, 0.10),
            prompt_length_range=(16, 256),
            output_length_range=(16, 128),
            priority_weights=_default_priority_weights(),
        ),
        "medium": WorkloadProfileConfig(
            name="medium",
            num_requests=500,
            arrival_interval_range=(0.01, 0.05),
            prompt_length_range=(32, 512),
            output_length_range=(32, 256),
            priority_weights=_default_priority_weights(),
        ),
        "heavy": WorkloadProfileConfig(
            name="heavy",
            num_requests=1000,
            arrival_interval_range=(0.002, 0.02),
            prompt_length_range=(64, 1024),
            output_length_range=(64, 512),
            priority_weights=_default_priority_weights(),
        ),
    }


@dataclass
class WorkloadConfig:
    """Configuration for workload generation and dataset management.

    Attributes:
        random_seed: Random seed for reproducible synthetic workload generation.
        dataset_dir: Directory where JSON workload traces are stored or generated.
        profiles: Dictionary of available workload profiles.
    """

    random_seed: int = 42
    dataset_dir: Path = field(default_factory=lambda: Path("workload/datasets"))
    profiles: Dict[str, WorkloadProfileConfig] = field(default_factory=_default_workload_profiles)

    def __post_init__(self) -> None:
        """Validate workload configuration.

        Raises:
            ValueError: If configuration values or profile types are invalid.
        """
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an integer, got {self.random_seed!r}")
        if not isinstance(self.dataset_dir, Path):
            if isinstance(self.dataset_dir, str):
                self.dataset_dir = Path(self.dataset_dir)
            else:
                raise ValueError(f"dataset_dir must be a Path or str, got {type(self.dataset_dir).__name__}")
        if not isinstance(self.profiles, dict):
            raise ValueError(f"profiles must be a dict, got {type(self.profiles).__name__}")
        for k, prof in self.profiles.items():
            if not isinstance(prof, WorkloadProfileConfig):
                raise ValueError(
                    f"profile '{k}' must be an instance of WorkloadProfileConfig, got {type(prof).__name__}"
                )

    def get_profile(self, name: str) -> WorkloadProfileConfig:
        """Retrieve a workload profile by name.

        Args:
            name: Name of the workload profile (e.g. 'light', 'medium', 'heavy').

        Returns:
            The corresponding WorkloadProfileConfig.

        Raises:
            ValueError: If the profile name is unknown.
        """
        if name not in self.profiles:
            available = sorted(list(self.profiles.keys()))
            raise ValueError(
                f"Unknown workload profile '{name}'. Available profiles: {available}"
            )
        return self.profiles[name]


@dataclass
class SimulationConfig:
    """Parameters governing the discrete-event simulation environment.

    Attributes:
        time_step: Simulation discrete time tick resolution in simulated seconds.
            Must be a positive, finite number.
        max_duration: Optional maximum simulated time in seconds after which the
            simulation stops. If ``None``, the simulation runs until all requests
            complete execution. If provided, must be a positive, finite number.
        results_dir: Directory path where raw simulation results, metrics, and
            reports are saved.
        logs_dir: Directory path where simulation logs are written.
    """

    time_step: float = 0.001
    max_duration: Optional[float] = None
    results_dir: Path = field(default_factory=lambda: Path("results"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self) -> None:
        """Validate simulation parameters.

        Raises:
            ValueError: If time_step or max_duration violate constraints.
        """
        if (
            isinstance(self.time_step, bool)
            or not isinstance(self.time_step, (int, float))
            or not math.isfinite(self.time_step)
            or self.time_step <= 0.0
        ):
            raise ValueError(f"time_step must be a positive finite float, got {self.time_step!r}")
        self.time_step = float(self.time_step)

        if self.max_duration is not None:
            if (
                isinstance(self.max_duration, bool)
                or not isinstance(self.max_duration, (int, float))
                or not math.isfinite(self.max_duration)
                or self.max_duration <= 0.0
            ):
                raise ValueError(
                    f"max_duration must be None or a positive finite float, got {self.max_duration!r}"
                )
            self.max_duration = float(self.max_duration)

        if not isinstance(self.results_dir, Path):
            if isinstance(self.results_dir, str):
                self.results_dir = Path(self.results_dir)
            else:
                raise ValueError(f"results_dir must be a Path or str, got {type(self.results_dir).__name__}")

        if not isinstance(self.logs_dir, Path):
            if isinstance(self.logs_dir, str):
                self.logs_dir = Path(self.logs_dir)
            else:
                raise ValueError(f"logs_dir must be a Path or str, got {type(self.logs_dir).__name__}")


@dataclass
class Config:
    """Master configuration container for the MoE serving simulation framework.

    Attributes:
        cluster: Cluster hardware and MoE model configuration.
        workload: Workload presets and synthetic request stream configuration.
        simulation: Discrete-event simulation parameters and output paths.
    """

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
