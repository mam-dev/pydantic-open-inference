from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import sys

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self

import pydantic

from ._client import OpenInferenceHTTPClientAPI
from ._utils import (
    OpenInferenceAPIInput,
    OpenInferenceAPIOutput,
    OpenInferenceAPIRequestedOutput,
    get_data,
    get_datatype,
    get_shape,
    is_listlike,
    unflatten_data,
)


class InputsBaseModel(pydantic.BaseModel):
    """Base class for model inputs.

    Subclass this class to define the model inputs, e.g.,

        class MyModelInput(InputsBaseModel):
            values: list[int]

    The field names MUST correspond to the names of the model inputs.

    The structure of a field must be mappable to the shape of the
    corresponding input, e.g.,
    - "list[tuple[int, int]]" for a shape of [-1, 2],
    - "tuple[float, float]" for  a shape of [2], or
    - "list[tuple[tuple[int, int], tuple[int, int]]]" for a shape of [-1, 2, 2].
    The actual shape in the input that is sent will have the actual size, not -1.

    NOTE: You can define an input field that is neither a list nor a tuple, just
    a simple datatype like int, float, str; this will correspond to a shape of [1].

    You can use typing.Namedtuple instead of a tuple to get named fields.

    Example:
        class Point(typing.Namedtuple):
            x: float
            y: float

        class ExampleInput(InputsBaseModel):
            points: list[Point]  # name: "points", shape: [-1, 2]
            matrix: tuple[tuple[int, int, int], tuple[int, int, int]]  # name: "matrix", shape: [2, 3]
            text: str  # name: "text", shape: [1]

    The datatype of an input is automatically set depending on the field type, e.g.,
    for a field that is "list[int]" (or "list[tuple[int, int]]" etc) the datatype
    will be "INT64". The actual mapping is

        bool: "BOOL",
        int: "INT64",
        float: "FP32",
        str: "BYTES",

    You can override these defaults using DatatypeOverride, e.g.,

        class ExampleInput(InputsBaseModel):
            points: list[Point]  # datatype: "INT64"
            more_points: Annotated[list[Point], DatatypeOverride("INT32")]  # datatype: "INT32"

    See DatatypeOverride docs for more information.

    NOTE: Because there is just one datatype per input, it means that fields with a mix of types
    like "tuple[int, float, str]" should be avoided.

    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, coerce_numbers_to_str=True)

    def to_inputs(self) -> list[OpenInferenceAPIInput]:
        return [
            {
                "name": field_name,
                "shape": get_shape(field_value),
                "datatype": get_datatype(field_value, type(self).model_fields[field_name]),
                "data": get_data(field_value),
            }
            for field_name, field_value in self.model_dump(mode="json", warnings=False).items()
        ]


class OutputsBaseModel(pydantic.BaseModel):
    """Base class for model outputs.

    Subclass this class to define the model inputs, e.g.,

        class MyModelIOutput(OutputsBaseModel):
            values: list[float]

    The field names MUST correspond to the names of the model outputs.

    The structure of a field must be mappable to the shape of the
    corresponding output, e.g.,
    - "list[tuple[int, int]]" for a shape of [-1, 2],
    - "tuple[float, float]" for  a shape of [2], or
    - "list[tuple[tuple[int, int], tuple[int, int]]]" for a shape of [-1, 2, 2].

    NOTE: You can define an output field that is neither a list nor a tuple, just
    a simple datatype like int, float, str; this will correspond to a shape of [1].

    You can use typing.Namedtuple instead of a tuple to get named fields.

    Example:
        class Point(typing.Namedtuple):
            x: float
            y: float

        class ExampleInput(OutputsBaseModel):
            points: list[Point]  # name: "points", shape: [-1, 2]
            matrix: tuple[tuple[int, int, int], tuple[int, int, int]]  # name: "matrix", shape: [2, 3]
            text: str  # name: "text", shape: [1]

    You can take advantage of pydantic's type coercion by defining the field types
    as you please, including mixed values.

    Example:
        Model outputs of the form
            [
                {
                    "name": "labels",
                    "datatype": "BYTES",
                    "shape": [3, 2],
                    "data": [
                        "SPAM",
                        "0.8",
                        "HAM",
                        "0.3",
                        "SPAM",
                        "0.9"
                    ]
                }
            ]

        can be, using this OutputsBaseModel:

            class LabelAndProbability(typing.Namedtuple):
                label: typing.Literal["SPAM", "HAM"]
                probability: float

            class ExampleInput(OutputsBaseModel):
                labels: list[LabelAndProbability]

        coerced to

            ExampleInput(
                labels=[
                    LabelAndProbability(label="SPAM", probability=0.8),
                    LabelAndProbability(label="HAM", probability=0.3),
                    LabelAndProbability(label="SPAM", probability=0.9),
                ]
            )

        i.e., we get named fields of the desired type.

    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, coerce_numbers_to_str=True)

    @classmethod
    def from_outputs(cls: type[Self], outputs: list[OpenInferenceAPIOutput]) -> Self:
        values: dict[str, Any] = {}
        for output in outputs:
            value = unflatten_data(output["shape"], output["data"])
            if output["shape"] == [1] and not is_listlike(cls.model_fields[output["name"]].annotation):
                values[output["name"]] = value[0]
            else:
                values[output["name"]] = value
        return cls(**values)

    @classmethod
    def get_requested_outputs(cls) -> list[OpenInferenceAPIRequestedOutput]:
        return [{"name": field_name} for field_name in cls.model_fields]


InputsModelT = TypeVar("InputsModelT", bound=InputsBaseModel)
OutputsModelT = TypeVar("OutputsModelT", bound=OutputsBaseModel)


class RemoteModel(Generic[InputsModelT, OutputsModelT]):
    """Interface toward a remote model called via the Open Inference protocol.

    Define the server, model, inputs, and outputs, then call
    the "infer" method to call the remote model, automatically
    handling conversion between the inputs/outputs models and
    the Open Inference protocol.

    This class is designed to be thread-safe.

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        inputs_model: type[InputsModelT],
        outputs_model: type[OutputsModelT],
        server_url: str,
        request_timeout_seconds: float | None = None,
        model_version: str | None = None,
    ) -> None:
        """Initialize a RemoteModel.

        Args:
            model_name: The name of the model in the inference server.
            inputs_model: InputsBaseModel subclass defining the model inputs.
            outputs_model: OutputsBaseModel subclass defining the model outputs.
            server_url: The URL of the inference server (not including the
                API version "v2/..." and onward.)
            request_timeout_seconds: Optional timeout (seconds) for requests to the inference server.
            model_version: Optional model version.

        """
        self._model_name = model_name
        self._model_version = model_version
        self._inputs_model = inputs_model
        self._outputs_model = outputs_model
        self._client_API = OpenInferenceHTTPClientAPI(base_url=server_url)
        self._request_timeout_seconds = request_timeout_seconds

    def infer(self, inputs: InputsModelT) -> OutputsModelT:
        """Use the remote model for inference.

        Args:
            inputs: The model inputs in the form of the InputsBaseModel
                subclass defined for this RemoteModel instance.

        Returns:
            outputs: The model outputs in the form of the OutputsBaseModel
                subclass defined for this RemoteModel instance.

        """
        if not isinstance(inputs, self._inputs_model):
            raise TypeError(f"Bad inputs type: {type(inputs)}")  # noqa: TRY003
        return self._outputs_model.from_outputs(
            self._client_API.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=inputs.to_inputs(),
                outputs=self._outputs_model.get_requested_outputs(),
                timeout_seconds=self._request_timeout_seconds,
            )
        )
