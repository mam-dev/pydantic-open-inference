"""Example: error handling."""

import sys

if sys.version_info < (3, 11):  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        """Replacement for enum.StrEnum, introduced in 3.11."""
else:  # pragma: no cover
    from enum import StrEnum

from pydantic import Field

from pydantic_open_inference import (
    BadStatusCodeFromServerError,
    InputsBaseModel,
    OutputsBaseModel,
    PydanticOpenInferenceError,
    RemoteModel,
)


class SpamLabel(StrEnum):
    """The possible values of the "label" output of the model."""

    SPAM = "spam"
    HAM = "ham"


class TextClassificationInput(InputsBaseModel):
    """Input for a text classification model."""

    text: str  # shape: [1], datatype: BYTES


class TextClassificationOutput(OutputsBaseModel):
    """Output from a text classification model."""

    label: SpamLabel  # The predicted label
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence score


text_classifier = RemoteModel(
    model_name="text_classifier",
    inputs_model=TextClassificationInput,
    outputs_model=TextClassificationOutput,
    server_url="http://localhost:8000",  # Your Triton server URL
    request_timeout_seconds=10.0,
)


try:
    input_data = TextClassificationInput(text="This is a great product!")
    result = text_classifier.infer(input_data)
    print(f"Result: {result}")
except BadStatusCodeFromServerError as error:
    print(f"Server responded with bad status code: HTTP{error.status_code}")
    # Handle the error (retry, log, fallback, etc.)
except PydanticOpenInferenceError as error:
    print(f"Inference failed: {error}")
    # Handle the error (retry, log, fallback, etc.)
except TypeError as error:
    print(f"Input type mismatch: {error}")
