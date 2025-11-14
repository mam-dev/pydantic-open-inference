from __future__ import annotations

import itertools
import sys
import types
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypedDict

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired  # pragma: no cover
else:
    from typing import NotRequired  # pragma: no cover

if sys.version_info < (3, 12):
    from typing_extensions import override  # pragma: no cover
else:
    from typing import override  # pragma: no cover

if sys.version_info < (3, 13):
    from more_itertools import batched  # pragma: no cover
else:
    from itertools import batched  # pragma: no cover
if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic.fields import FieldInfo

Shape = list[int]
Datatype = Literal[
    "BOOL",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "FP16",
    "FP32",
    "FP64",
    "BYTES",
]
Data = list[Any]
_TYPE_TO_DATATYPE_MAP: Final[Mapping[type, Datatype]] = {
    bool: "BOOL",
    int: "INT64",
    float: "FP32",
    str: "BYTES",
}


class PydanticOpenInferenceError(Exception):
    """Package base exception."""


class ShapeDataMismatchError(PydanticOpenInferenceError):
    """Exception raised when shape and data do not match."""


class DatatypeOverride:
    """Use with typing.Annotated to override the default datatype in inputs.

    In this example InputsBaseModel, the datatype of
    "values" would be "INT64":

        class IntInputsBaseModel(InputsBaseModel):
            values: list[int]

    Using DatatypeOverride, we can instead force the datatype
    to be anything we want, e.g., "INT16":

        class IntInputsBaseModel(InputsBaseModel):
            values: Annotated[list[int], DatatypeOverride("INT16")]

    Note that this simply sets the datatype as given. There
    are no additional checks (for sign, size, etc).

    """

    __slots__ = ("_datatype",)

    def __init__(self, datatype: Datatype) -> None:
        self._datatype = datatype

    @property
    def datatype(self) -> Datatype:
        return self._datatype


def is_flat(data: Data) -> bool:
    return not data or not any(isinstance(x, (list, tuple)) for x in data)


def parse_row_major_order(shape: Shape, data: list[Any]) -> list[Any]:
    if len(shape) == 1:
        if shape[0] != len(data):
            raise ShapeDataMismatchError
        return data
    new_shape: Shape = list(shape[:-1])
    return parse_row_major_order(new_shape, list(batched(data, n=shape[-1], strict=True)))


def is_listlike(annotation: type[Any] | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, types.GenericAlias):
        annotation = annotation.__origin__  # type: ignore[unreachable]
    return any(t in (list, tuple) for t in itertools.chain((annotation,), annotation.__bases__))


def unflatten_data(shape: Shape, data: Data) -> Data:
    if is_flat(data):
        return parse_row_major_order(shape, data)
    return data


def get_shape(value: Any) -> Shape:
    shape: Shape = []
    while isinstance(value, (list, tuple)):
        shape.append(len(value))
        value = value[0]
    return shape or [1]


def get_datatype(value: Any, field_info: FieldInfo) -> Datatype:
    overrides: list[DatatypeOverride] = [x for x in field_info.metadata if isinstance(x, DatatypeOverride)]
    if overrides:
        return overrides[0].datatype
    while isinstance(value, (list, tuple)):
        value = value[0]
    return _TYPE_TO_DATATYPE_MAP[type(value)]


def get_data(value: Any) -> Data:
    if not isinstance(value, (tuple, list)):
        return [value]
    return list(value)


class _OpenInferenceAPIPut(TypedDict):
    name: str
    shape: Shape
    datatype: Datatype
    data: Data


class OpenInferenceAPIInput(_OpenInferenceAPIPut):
    parameters: NotRequired[dict[str, Any]]


class OpenInferenceAPIRequestedOutput(TypedDict):
    name: str
    parameters: NotRequired[dict[str, Any]]


class OpenInferenceAPIOutput(_OpenInferenceAPIPut):
    parameters: NotRequired[dict[str, Any]]


class Singleton(type):
    _instances: ClassVar[dict[tuple[type, tuple[Any, ...], str], Singleton]] = {}

    @override
    def __call__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        key = (cls, args, str(kwargs))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]
