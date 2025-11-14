"""Example: Call a model with multiple inputs and outputs."""

import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        """Replacement for enum.StrEnum, introduced in 3.11."""
else:
    from enum import StrEnum

from pydantic import Field

from pydantic_open_inference import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)


class MultiModalInput(InputsBaseModel):
    """Multiple inputs for a multi-modal model."""

    text: str
    # We could use a 10-tuple for numerical_features, but it might be unwieldy,
    # so we use constrained list instead. It is a matter of taste.
    numerical_features: list[float] = Field(min_length=10, max_length=10)  # Shape: [10]
    categorical_ids: tuple[int, int, int, int, int]  # Shape: [5]


class Classification(StrEnum):
    """The possible classifications."""

    A = "A"
    B = "B"
    C = "C"


class MultiModalOutput(OutputsBaseModel):
    """Multiple outputs from the model."""

    classification: Classification
    # Here we add some extra validation that probability is in the interval [0.0, 1.0],
    # but we don't *have to*; it still works, of course.
    probability: float = Field(ge=0.0, le=1.0)
    attention_weights: list[float]  # Shape: [-1]


multimodal_model = RemoteModel(
    model_name="multimodal_classifier",
    inputs_model=MultiModalInput,
    outputs_model=MultiModalOutput,
    server_url="http://localhost:8000",
    request_timeout_seconds=30.0,  # Longer timeout for complex models
)

multi_input = MultiModalInput(
    text="Customer feedback about service quality",
    numerical_features=[0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6, 0.7, 0.15],
    categorical_ids=(1, 5, 3, 7, 2),
)

multi_result = multimodal_model.infer(multi_input)
print(f"Classification: {multi_result.classification}")
print(f"Probability: {multi_result.probability:.3f}")
print(f"Attention weights: {multi_result.attention_weights[:5]}...")  # First 5
