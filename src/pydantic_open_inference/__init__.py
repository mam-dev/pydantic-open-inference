"""Package for wrapping calls to an Open Inference server REST API."""

from ._client import BadStatusCodeFromServerError
from ._remote_model import InputsBaseModel, OutputsBaseModel, RemoteModel
from ._utils import DatatypeOverride, IncompatibleTensorError, PydanticOpenInferenceError

__all__ = (
    "BadStatusCodeFromServerError",
    "DatatypeOverride",
    "IncompatibleTensorError",
    "InputsBaseModel",
    "OutputsBaseModel",
    "PydanticOpenInferenceError",
    "RemoteModel",
)
