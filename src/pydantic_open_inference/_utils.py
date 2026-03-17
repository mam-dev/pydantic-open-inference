from __future__ import annotations

import itertools
import sys
import types
from typing import TYPE_CHECKING, Any, ClassVar, Final, ForwardRef, Literal, TypedDict, get_args, get_origin

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, Self  # pragma: no cover
else:
    from typing import NotRequired, Self  # pragma: no cover

if sys.version_info < (3, 12):
    from typing_extensions import override  # pragma: no cover
else:
    from typing import override  # pragma: no cover

if sys.version_info < (3, 13):
    from more_itertools import batched  # pragma: no cover
else:
    from itertools import batched  # pragma: no cover

if sys.version_info >= (3, 14):  # pragma: no cover
    from typing import evaluate_forward_ref
elif sys.version_info >= (3, 13):  # pragma: no cover

    def evaluate_forward_ref(
        forward_ref: ForwardRef,
        *,
        type_params: Any,
        globals: Any | None = None,  # noqa: A002
        locals: Any | None = None,  # noqa: A002
        **__kwargs: Any,
    ) -> Any | None:
        return forward_ref._evaluate(globals, locals, type_params=type_params, recursive_guard=frozenset())  # noqa: SLF001
else:  # pragma: no cover

    def evaluate_forward_ref(
        forward_ref: ForwardRef,
        *,
        globals: Any | None = None,  # noqa: A002
        locals: Any | None = None,  # noqa: A002
        **__kwargs: Any,
    ) -> Any | None:
        return forward_ref._evaluate(globals, locals, recursive_guard=frozenset())  # noqa: SLF001


if TYPE_CHECKING:
    from collections.abc import Mapping

    import pydantic
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


def get_datatype(value_or_type: Any, field_info: FieldInfo) -> Datatype:
    overrides: list[DatatypeOverride] = [x for x in field_info.metadata if isinstance(x, DatatypeOverride)]
    if overrides:
        return overrides[0].datatype
    while isinstance(value_or_type, (list, tuple)):
        value_or_type = value_or_type[0]
    if not isinstance(value_or_type, type):
        value_or_type = type(value_or_type)
    return _TYPE_TO_DATATYPE_MAP[value_or_type]


def get_data(value: Any) -> Data:
    if not isinstance(value, (tuple, list)):
        return [value]
    return list(value)


class OpenInferenceMetadataTensor(TypedDict):
    name: str
    shape: Shape
    datatype: Datatype


class _OpenInferenceAPIPut(OpenInferenceMetadataTensor):
    data: Data


class OpenInferenceAPIInput(_OpenInferenceAPIPut):
    parameters: NotRequired[dict[str, Any]]


class OpenInferenceAPIRequestedOutput(TypedDict):
    name: str
    parameters: NotRequired[dict[str, Any]]


class OpenInferenceAPIOutput(_OpenInferenceAPIPut):
    parameters: NotRequired[dict[str, Any]]


class OpenInferenceModelMetadata(TypedDict):
    name: str
    versions: NotRequired[list[str] | None]
    platform: str
    inputs: list[OpenInferenceMetadataTensor]
    outputs: list[OpenInferenceMetadataTensor]


class Singleton(type):
    _instances: ClassVar[dict[tuple[type, tuple[Any, ...], str], Singleton]] = {}

    @override
    def __call__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        key = (cls, args, str(kwargs))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


class IncompatibleTensorError(PydanticOpenInferenceError):
    """Raised when input/output of a model is incompatible with its model definition."""

    @classmethod
    def for_datatype_mismatch(cls, name: str, local_datatype: Datatype, remote_datatype: Datatype) -> Self:
        return cls(f"Datatype mismatch for {name}, {local_datatype} (local) != {remote_datatype} (remote)")

    @classmethod
    def for_shape_mismatch(cls, name: str, local_shape: Shape, remote_shape: Shape) -> Self:
        return cls(f"Shape mismatch for {name}, {local_shape} (local) != {remote_shape} (remote)")

    @classmethod
    def for_missing_input(cls, name: str) -> Self:
        return cls(f"Model does not have an input named {name}")

    @classmethod
    def for_missing_output(cls, name: str) -> Self:
        return cls(f"Model does not have an output named {name}")


_SIMPLE_TYPES = (bool, str, bytes, int, float)


def unnest_type(field_type: Any) -> list[Any]:
    unnested: list[Any] = []
    while field_type is not None:
        if (origin := get_origin(field_type)) is not None:
            unnested.append(origin)
            args = get_args(field_type)
            field_type = args[0] if len(args) == 1 else args
        elif isinstance(field_type, tuple):
            unnested.append(field_type)
            field_type = None
        elif issubclass(field_type, tuple) and hasattr(field_type, "__annotations__"):
            inner_types = tuple(
                evaluate_forward_ref(
                    field, type_params=getattr(field_type, "__type_params__", None), globals=globals(), locals=locals()
                )
                if isinstance(field, ForwardRef)
                else field
                for field in field_type.__annotations__.values()
            )
            field_type = inner_types
        else:
            unnested.append(field_type)
            field_type = None
    return unnested


def get_allowed_shape_of_type(field_type: Any, *, root_level: bool) -> Shape:  # noqa: PLR0911
    if isinstance(field_type, tuple):
        return [len(field_type)]
    origin = get_origin(field_type)
    if isinstance(field_type, types.GenericAlias) and issubclass(origin, list):
        return [-1]
    if isinstance(field_type, types.GenericAlias) and issubclass(origin, tuple):
        return [len(get_args(field_type))]
    if origin is Literal:
        return [1]
    if issubclass(field_type, list):
        return [-1]
    if issubclass(field_type, tuple) and hasattr(field_type, "_fields"):  # NameTuple/namedtuple
        return [len(field_type._fields)]
    if issubclass(field_type, tuple):
        return [-1]
    if issubclass(field_type, _SIMPLE_TYPES) and root_level:
        return [1]
    if issubclass(field_type, _SIMPLE_TYPES) and not root_level:
        return []
    raise PydanticOpenInferenceError(f"Unsupported field_type: {type(field_type)}")  # noqa: TRY003  # pragma: no cover


def get_input_tensor_by_name(name: str, metadata: OpenInferenceModelMetadata) -> OpenInferenceMetadataTensor:
    try:
        return _get_put_tensor_by_name(name, metadata, "inputs")
    except KeyError as missing:
        raise IncompatibleTensorError.for_missing_input(str(missing)) from None


def get_output_tensor_by_name(name: str, metadata: OpenInferenceModelMetadata) -> OpenInferenceMetadataTensor:
    try:
        return _get_put_tensor_by_name(name, metadata, "outputs")
    except KeyError as missing:
        raise IncompatibleTensorError.for_missing_output(str(missing)) from None


def _get_put_tensor_by_name(
    name: str, metadata: OpenInferenceModelMetadata, put_type: Literal["inputs", "outputs"]
) -> OpenInferenceMetadataTensor:
    for tensor in metadata[put_type]:
        if tensor["name"] == name:
            return tensor
    raise KeyError(name)


def validate_model_tensor(
    model_tensor: OpenInferenceMetadataTensor,
    local_field: pydantic.fields.FieldInfo,
    *,
    is_input: bool,
) -> None:
    unnested = unnest_type(local_field.annotation)

    local_shape: Shape = []
    for index, type_value in enumerate(unnested):
        local_shape.extend(get_allowed_shape_of_type(type_value, root_level=index == 0))
    remote_shape = model_tensor["shape"]
    # Local shape element can be anything if the corresponding remote shape element is -1,
    # otherwise they have to match exactly.
    if local_shape != remote_shape and (
        len(local_shape) != len(remote_shape)
        or any(rm not in (loc, -1) for (rm, loc) in zip(remote_shape, local_shape, strict=True))
    ):
        raise IncompatibleTensorError.for_shape_mismatch(
            name=model_tensor["name"], local_shape=local_shape, remote_shape=remote_shape
        )

    if is_input:
        # The datatype we output must match the one expected by the remote
        local_datatype = get_datatype(unnested[-1], local_field)
        remote_datatype = model_tensor["datatype"]
        if local_datatype != remote_datatype:
            raise IncompatibleTensorError.for_datatype_mismatch(
                name=model_tensor["name"], local_datatype=local_datatype, remote_datatype=remote_datatype
            )
