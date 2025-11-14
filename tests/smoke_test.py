"""Smoke tests to run on wheel/sdist before publishing.

Just checks that imports etc work properly.

"""

from pydantic_open_inference import (
    BadStatusCodeFromServerError,  # noqa: F401
    DatatypeOverride,  # noqa: F401
    InputsBaseModel,
    OutputsBaseModel,
    PydanticOpenInferenceError,  # noqa: F401
    RemoteModel,
)


class SmokeTestInputs(InputsBaseModel):
    dummy: str


class SmokeTestOutputs(OutputsBaseModel):
    dummy: str


_ = RemoteModel(
    model_name="dummy",
    inputs_model=SmokeTestInputs,
    outputs_model=OutputsBaseModel,
    server_url="http://localhost:8080",
)
