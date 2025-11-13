"""Package for wrapping calls to an Open Inference server REST API."""

from ._client import BadStatusCodeFromServerError
from ._remote_model import InputsBaseModel, OutputsBaseModel, RemoteModel
from ._utils import DatatypeOverride, PydanticOpenInferenceError

__all__ = (
    "BadStatusCodeFromServerError",
    "DatatypeOverride",
    "InputsBaseModel",
    "OutputsBaseModel",
    "PydanticOpenInferenceError",
    "RemoteModel",
)
