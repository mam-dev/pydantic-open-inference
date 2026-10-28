"""Example: Call a text classification model residing in an open-inference server.

This is a simple example of calling a model with a single "text" input
and two outputs named "label" and "confidence".

Here we add more constraints in our pydantic models to achieve stricter
validation and more useful typing. We use enum.StrEnum for the label,
corresponding to the possible labels output by the model (using
typing.Literal is another option) and add validation that the confidence
is between 0.0 and 1.0.

"""

from enum import StrEnum

from pydantic import Field

from pydantic_open_inference import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)


class SpamLabel(StrEnum):
    """The possible values of the "label" output of the model."""

    SPAM = "spam"
    HAM = "ham"


class TextClassificationInput(InputsBaseModel):
    """Input for the text classification model."""

    text: str  # shape: [1], datatype: BYTES


class TextClassificationOutput(OutputsBaseModel):
    """Output from the text classification model."""

    label: SpamLabel  # The predicted label, shape: [1]
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence score, shape: [1]


# Create the remote model client
text_classifier = RemoteModel(
    model_name="text_classifier",
    inputs_model=TextClassificationInput,
    outputs_model=TextClassificationOutput,
    server_url="http://localhost:8000",  # Your Triton server URL
    request_timeout_seconds=10.0,
)


input_data = TextClassificationInput(text="This is a great product!")
result = text_classifier.infer(input_data)

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence}")
