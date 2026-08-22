"""Inference request model.

Defines the structure of a single simulated LLM inference request, including
its arrival time, prompt/output token counts, priority level, and the runtime
bookkeeping fields populated by the scheduler and simulation environment.

All timestamps (``arrival_time``, ``start_time``, ``first_token_time``,
``completion_time``) represent **simulated time**, not actual wall-clock time.
"""

from dataclasses import dataclass
from typing import Optional

#: Priority levels supported by the scheduling framework.
VALID_PRIORITIES: tuple = ("low", "normal", "high")


@dataclass
class InferenceRequest:
    """A single inference request flowing through the simulated MoE serving system.

    The core fields describe *what* the request is; they are provided at
    construction time and validated immediately. The runtime fields are
    placeholders that the scheduler and simulation environment fill in later
    as the request moves through the system.

    Attributes:
        request_id: Unique identifier for the request. Must be non-negative.
        arrival_time: Simulated time at which the request enters the system.
            Must be greater than or equal to 0.
        prompt_length: Number of input tokens in the prompt. Must be > 0.
        output_length: Expected number of output tokens to generate.
            Must be > 0.
        priority: Scheduling priority. One of ``"low"``, ``"normal"``,
            or ``"high"`` (exact lowercase match).
        assigned_gpu: GPU ID selected by the scheduler, or ``None`` if the
            request has not been dispatched yet.
        queue_wait_time: Simulated duration the request spent waiting in a
            queue before execution began. Defaults to 0.0.
        start_time: Simulated time at which processing begins, or ``None``
            if the request has not started yet.
        first_token_time: Simulated time at which the first output token was
            generated (used for TTFT metrics), or ``None`` until it happens.
        completion_time: Simulated time at which the request finished
            generating all output tokens, or ``None`` while incomplete.
    """

    # --- Core fields (required) -------------------------------------------
    request_id: int
    arrival_time: float
    prompt_length: int
    output_length: int
    priority: str

    # --- Runtime fields (populated by scheduler / environment) -------------
    assigned_gpu: Optional[int] = None
    queue_wait_time: float = 0.0
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate core fields after construction.

        Raises:
            ValueError: If any core field violates its constraints.
        """
        if self.request_id < 0:
            raise ValueError(
                f"request_id must be non-negative, got {self.request_id}"
            )
        if self.arrival_time < 0:
            raise ValueError(
                f"arrival_time must be >= 0, got {self.arrival_time}"
            )
        if self.prompt_length <= 0:
            raise ValueError(
                f"prompt_length must be > 0, got {self.prompt_length}"
            )
        if self.output_length <= 0:
            raise ValueError(
                f"output_length must be > 0, got {self.output_length}"
            )
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {VALID_PRIORITIES}, "
                f"got '{self.priority}'"
            )

    def total_tokens(self) -> int:
        """Return the total number of tokens this request will process.

        Returns:
            The sum of ``prompt_length`` and ``output_length``.
        """
        return self.prompt_length + self.output_length

    def is_completed(self) -> bool:
        """Check whether the request has finished processing.

        Returns:
            ``True`` if ``completion_time`` has been set, ``False`` otherwise.
        """
        return self.completion_time is not None
