"""Example: Call a text classification model residing in an open-inference server.

This is a simple example of calling a model with a single "text" input
and two outputs named "label" and "confidence".

"""

from pydantic_open_inference import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)


class TextClassificationInput(InputsBaseModel):
    """Input for the text classification model."""

    text: str  # shape: [1], datatype: BYTES


class TextClassificationOutput(OutputsBaseModel):
    """Output from the text classification model."""

    label: str  # The predicted label, shape: [1]
    confidence: float  # Confidence score, shape [1]


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
