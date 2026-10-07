"""Workload generator for LLM inference requests.

Provides the :class:`WorkloadGenerator` class for generating synthetic streams
of :class:`~simulation.request.InferenceRequest` instances based on configured
workload profiles (e.g. light, medium, heavy).
"""

import random
from typing import List, Optional

from config.config import WorkloadConfig, WorkloadProfileConfig
from simulation.request import InferenceRequest


class WorkloadGenerator:
    """Generates synthetic streams of LLM inference requests for MoE serving simulation.

    Uses an isolated pseudo-random number generator (:class:`random.Random`) initialized
    with the configured random seed to ensure deterministic, reproducible request generation
    without affecting the global random state.

    Attributes:
        config: The workload configuration providing seeds, directories, and profiles.
        profile_name: The default profile name used for request generation.
    """

    def __init__(
        self,
        config: Optional[WorkloadConfig] = None,
        profile_name: str = "medium",
    ) -> None:
        """Initialize the workload generator.

        Args:
            config: Optional workload configuration. If ``None``, a default
                :class:`WorkloadConfig` is instantiated.
            profile_name: Name of the workload profile to select (e.g. 'light',
                'medium', 'heavy'). Defaults to 'medium'.

        Raises:
            ValueError: If ``config`` is not a :class:`WorkloadConfig` instance
                or if ``profile_name`` is not found in ``config.profiles``.
        """
        if config is None:
            config = WorkloadConfig()
        elif not isinstance(config, WorkloadConfig):
            raise ValueError(
                f"config must be an instance of WorkloadConfig, got {type(config).__name__}"
            )

        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError(
                f"profile_name must be a non-empty string, got {profile_name!r}"
            )

        if profile_name not in config.profiles:
            available = sorted(list(config.profiles.keys()))
            raise ValueError(
                f"Unknown workload profile '{profile_name}'. Available profiles: {available}"
            )

        self.config: WorkloadConfig = config
        self.profile_name: str = profile_name
        self._rng: random.Random = random.Random(self.config.random_seed)

    def reset(self) -> None:
        """Reset the internal PRNG stream to the configured random seed."""
        self._rng.seed(self.config.random_seed)

    def generate(
        self,
        profile_name: Optional[str] = None,
        reset_seed: bool = True,
    ) -> List[InferenceRequest]:
        """Generate a synthetic stream of inference requests.

        Args:
            profile_name: Optional name of the profile to generate. If ``None``,
                the generator's default ``profile_name`` is used.
            reset_seed: If ``True``, resets the internal random generator to
                ``config.random_seed`` before generating, ensuring identical
                repeatable batches. If ``False``, continues the current random
                stream to produce consecutive distinct batches. In both cases,
                the returned batch starts at request ID 0 and arrival time 0.0.

        Returns:
            A list of :class:`InferenceRequest` instances with sequential IDs starting
            at 0, arrival times starting at 0.0 (simulated seconds), and sampled
            token lengths and priorities.

        Raises:
            ValueError: If ``profile_name`` is provided but not found in ``config.profiles``.
        """
        target_profile_name = self.profile_name if profile_name is None else profile_name
        if not isinstance(target_profile_name, str) or target_profile_name not in self.config.profiles:
            available = sorted(list(self.config.profiles.keys()))
            raise ValueError(
                f"Unknown workload profile '{target_profile_name}'. Available profiles: {available}"
            )

        profile: WorkloadProfileConfig = self.config.get_profile(target_profile_name)

        if reset_seed:
            self.reset()

        requests: List[InferenceRequest] = []
        current_arrival_time: float = 0.0

        min_arr, max_arr = profile.arrival_interval_range
        min_prompt, max_prompt = profile.prompt_length_range
        min_output, max_output = profile.output_length_range

        priorities = list(profile.priority_weights.keys())
        priority_weights = list(profile.priority_weights.values())

        for req_id in range(profile.num_requests):
            if req_id > 0:
                interval = self._rng.uniform(min_arr, max_arr)
                current_arrival_time += interval

            prompt_len = self._rng.randint(min_prompt, max_prompt)
            output_len = self._rng.randint(min_output, max_output)
            priority = self._rng.choices(priorities, weights=priority_weights, k=1)[0]

            request = InferenceRequest(
                request_id=req_id,
                arrival_time=current_arrival_time,
                prompt_length=prompt_len,
                output_length=output_len,
                priority=priority,
            )
            requests.append(request)

        return requests
