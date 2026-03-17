"""Tests for the _utils module."""

from __future__ import annotations

import contextlib
import types
from typing import Annotated, Any, Literal, NamedTuple
from unittest.mock import Mock

import pydantic
import pytest
from pydantic.fields import FieldInfo

from pydantic_open_inference._utils import (
    Data,
    Datatype,
    DatatypeOverride,
    IncompatibleTensorError,
    OpenInferenceMetadataTensor,
    OpenInferenceModelMetadata,
    Shape,
    ShapeDataMismatchError,
    Singleton,
    get_allowed_shape_of_type,
    get_data,
    get_datatype,
    get_input_tensor_by_name,
    get_output_tensor_by_name,
    get_shape,
    is_flat,
    is_listlike,
    parse_row_major_order,
    unflatten_data,
    unnest_type,
    validate_model_tensor,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    Singleton._instances.clear()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], True),
        ([1, 2, 3, 4], True),
        ([[1, 2], [3, 4]], False),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], False),
    ],
)
def test_is_flat(values: list[Any], expected: bool) -> None:
    assert is_flat(values) is expected


@pytest.mark.parametrize(
    "shape, values, expected",
    [
        pytest.param([1], ["hello world"], ["hello world"], id="single"),
        pytest.param([2], ["hello", "world"], ["hello", "world"], id="simple"),
        pytest.param([2, 2], [1, 2, 3, 4], [(1, 2), (3, 4)], id="2x2"),
        pytest.param(
            [2, 4],
            [0, 1, 2, 3, 10, 11, 12, 13],
            [(0, 1, 2, 3), (10, 11, 12, 13)],
            id="2x4",
        ),
        pytest.param(
            [3, 3, 2],
            [
                111,
                112,
                121,
                122,
                131,
                132,
                211,
                212,
                221,
                222,
                231,
                232,
                311,
                312,
                321,
                322,
                331,
                332,
            ],
            [
                ((111, 112), (121, 122), (131, 132)),
                ((211, 212), (221, 222), (231, 232)),
                ((311, 312), (321, 322), (331, 332)),
            ],
            id="3x3x2",
        ),
        pytest.param([1, 1], ["hello world"], [("hello world",)], id="1x1"),
    ],
)
def test_parse_row_major_order(shape: Shape, values: Any, expected: Any) -> None:
    assert parse_row_major_order(shape=shape, data=values) == expected


@pytest.mark.parametrize(
    "shape, values",
    [
        ([3], ["hello", "world"]),
    ],
)
def test_parse_row_major_order__error(shape: Shape, values: Any) -> None:
    with pytest.raises(ShapeDataMismatchError):
        _ = parse_row_major_order(shape=shape, data=values)


class ListSubClass(list[Any]): ...


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (None, False),
        (list, True),
        (tuple, True),
        (int, False),
        (float, False),
        (str, False),
        (dict, False),
        (types.GenericAlias(dict, (str, int)), False),
        (types.GenericAlias(list, (int,)), True),
        (ListSubClass, True),
    ],
)
def test_is_listlike(annotation: type[Any], expected: bool) -> None:
    assert is_listlike(annotation) is expected


@pytest.fixture
def mock_is_flat(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "pydantic_open_inference._utils.is_flat",
        mock_obj := Mock(spec=is_flat),
    )
    return mock_obj


@pytest.fixture
def mock_parse_row_major_order(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "pydantic_open_inference._utils.parse_row_major_order",
        mock_obj := Mock(spec=parse_row_major_order),
    )
    return mock_obj


def test_unflatten_data__flat(mock_is_flat: Mock, mock_parse_row_major_order: Mock) -> None:
    mock_is_flat.return_value = True
    shape = [2, 2]
    data = [1, 2, 3, 4]
    actual = unflatten_data(shape, data)
    mock_is_flat.assert_called_once_with(data)
    mock_parse_row_major_order.assert_called_once_with(shape, data)
    assert actual is mock_parse_row_major_order.return_value


def test_unflatten_data__nested(mock_is_flat: Mock, mock_parse_row_major_order: Mock) -> None:
    mock_is_flat.return_value = False
    shape = [2, 2]
    data = [[1, 2], [3, 4]]
    actual = unflatten_data(shape, data)
    mock_is_flat.assert_called_once_with(data)
    mock_parse_row_major_order.assert_not_called()
    assert actual is data


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hello world", [1]),
        (["hello", "world"], [2]),
        ([[1, 2], [3, 4], [5, 6]], [3, 2]),
    ],
)
def test_get_shape(value: Any, expected: Shape) -> None:
    assert get_shape(value) == expected


@pytest.mark.parametrize(
    "value, field_info, expected",
    [
        ("hello world", FieldInfo(), "BYTES"),
        (["hello", "world"], FieldInfo(), "BYTES"),
        ([[1, 2], [3, 4], [5, 6]], FieldInfo(), "INT64"),
        (
            [[1, 2], [3, 4], [5, 6]],
            Mock(spec=FieldInfo, metadata=[DatatypeOverride("INT16")]),
            "INT16",
        ),
        (str, FieldInfo(), "BYTES"),
        ([int], FieldInfo(), "INT64"),
    ],
)
def test_get_datatype(value: Any, field_info: FieldInfo, expected: Datatype) -> None:
    assert get_datatype(value, field_info) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hello world", ["hello world"]),
        (["hello", "world"], ["hello", "world"]),
        ([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]),
    ],
)
def test_get_data(value: Any, expected: Data) -> None:
    assert get_data(value) == expected


def test_singleton() -> None:
    class MySingleton(metaclass=Singleton):
        def __init__(self, name: str) -> None: ...

    instance_0 = MySingleton(name="A")
    instance_1 = MySingleton(name="A")
    instance_2 = MySingleton(name="B")
    assert instance_0 is instance_1
    assert instance_0 is not instance_2


class _PersonTuple(NamedTuple):
    name: str
    age: int


class _TextInput(pydantic.BaseModel):
    text: str


class _Entity(NamedTuple):
    score: float
    label: Literal["tracking-id", "order-id"]
    start: int
    end: int


class _EntityOutput(pydantic.BaseModel):
    entities: list[_Entity]
    overridden: Annotated[float, DatatypeOverride("FP16")]


@pytest.mark.parametrize(
    "field_type, expected",
    [
        (str, [str]),
        (int, [int]),
        (types.GenericAlias(list, (int,)), [list, int]),
        (types.GenericAlias(list, (types.GenericAlias(tuple, (str, int)),)), [list, tuple, (str, int)]),
        (_PersonTuple, [(str, int)]),
        (_TextInput.model_fields["text"].annotation, [str]),
        (
            _EntityOutput.model_fields["entities"].annotation,
            [list, (float, Literal["tracking-id", "order-id"], int, int)],
        ),
    ],
)
def test_unnest_type(field_type: type, expected: list[type | tuple[type, ...]]) -> None:
    assert unnest_type(field_type) == expected


@pytest.mark.parametrize(
    "field_type, root_level, expected",
    [
        (str, False, []),
        (str, True, [1]),
        (list, False, [-1]),
        (list, True, [-1]),
        (
            (str, float, str),
            False,
            [3],
        ),
        (
            types.GenericAlias(list, (int,)),
            True,
            [-1],
        ),
        (
            types.GenericAlias(list, (str,)),
            False,
            [-1],
        ),
        (
            types.GenericAlias(tuple, (int, float, str)),
            True,
            [3],
        ),
        (
            tuple,
            True,
            [-1],
        ),
        (
            _PersonTuple,
            True,
            [2],
        ),
        (
            Literal["tracking-id", "order-id"],
            False,
            [1],
        ),
    ],
)
def test_get_allowed_shape_of_type(field_type: type | tuple[type], root_level: bool, expected: Shape) -> None:
    assert get_allowed_shape_of_type(field_type, root_level=root_level) == expected


@pytest.mark.parametrize(
    "name, model_metadata, expected",
    [
        (
            "text",
            {"name": "ensemble", "platform": "triton", "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1]}]},
            {"name": "text", "datatype": "BYTES", "shape": [1]},
        ),
        (
            "text",
            {
                "name": "ensemble",
                "platform": "triton",
                "inputs": [
                    {"name": "stuff", "datatype": "FP32", "shape": [3]},
                    {"name": "text", "datatype": "BYTES", "shape": [1]},
                ],
            },
            {"name": "text", "datatype": "BYTES", "shape": [1]},
        ),
        (
            "text",
            {"name": "ensemble", "platform": "triton", "inputs": []},
            IncompatibleTensorError("Model does not have an input named 'text'"),
        ),
    ],
)
def test_get_input_tensor_by_name(
    name: str,
    model_metadata: OpenInferenceModelMetadata,
    expected: OpenInferenceMetadataTensor | IncompatibleTensorError,
) -> None:
    if isinstance(expected, IncompatibleTensorError):
        with pytest.raises(type(expected), match=str(expected)):
            _ = get_input_tensor_by_name(name, model_metadata)
    else:
        assert get_input_tensor_by_name(name, model_metadata) == expected


@pytest.mark.parametrize(
    "name, model_metadata, expected",
    [
        (
            "text",
            {
                "name": "ensemble",
                "platform": "triton",
                "outputs": [{"name": "text", "datatype": "BYTES", "shape": [1]}],
            },
            {"name": "text", "datatype": "BYTES", "shape": [1]},
        ),
        (
            "text",
            {
                "name": "ensemble",
                "platform": "triton",
                "outputs": [
                    {"name": "stuff", "datatype": "FP32", "shape": [3]},
                    {"name": "text", "datatype": "BYTES", "shape": [1]},
                ],
            },
            {"name": "text", "datatype": "BYTES", "shape": [1]},
        ),
        (
            "text",
            {"name": "ensemble", "platform": "triton", "outputs": []},
            IncompatibleTensorError("Model does not have an output named 'text'"),
        ),
    ],
)
def test_get_output_tensor_by_name(
    name: str,
    model_metadata: OpenInferenceModelMetadata,
    expected: OpenInferenceMetadataTensor | IncompatibleTensorError,
) -> None:
    if isinstance(expected, IncompatibleTensorError):
        with pytest.raises(type(expected), match=str(expected)):
            _ = get_output_tensor_by_name(name, model_metadata)
    else:
        assert get_output_tensor_by_name(name, model_metadata) == expected


@pytest.mark.parametrize(
    "model_tensor, local_field, is_input, expected_error_message",
    [
        ({"name": "text", "datatype": "BYTES", "shape": [1]}, _TextInput.model_fields["text"], True, None),
        (
            {"name": "entities", "datatype": "BYTES", "shape": [-1, 4]},
            _EntityOutput.model_fields["entities"],
            False,
            None,
        ),
        (
            {"name": "text", "datatype": "BYTES", "shape": [2]},
            _TextInput.model_fields["text"],
            True,
            r"Shape mismatch for text, \[1\] \(local\) != \[2\] \(remote\)",
        ),
        (
            {"name": "entities", "datatype": "BYTES", "shape": [3, 4]},
            _EntityOutput.model_fields["entities"],
            False,
            r"Shape mismatch for entities, \[-1, 4\] \(local\) != \[3, 4\] \(remote\)",
        ),
        (
            {"name": "entities", "datatype": "BYTES", "shape": [-1, 4, 2]},
            _EntityOutput.model_fields["entities"],
            False,
            r"Shape mismatch for entities, \[-1, 4\] \(local\) != \[-1, 4, 2\] \(remote\)",
        ),
        (
            {"name": "text", "datatype": "FP64", "shape": [1]},
            _TextInput.model_fields["text"],
            True,
            r"Datatype mismatch for text, BYTES \(local\) != FP64 \(remote\)",
        ),
        (
            {"name": "overridden", "datatype": "FP16", "shape": [1]},
            _EntityOutput.model_fields["overridden"],
            True,
            None,
        ),
        (
            {"name": "overridden", "datatype": "FP32", "shape": [1]},
            _EntityOutput.model_fields["overridden"],
            True,
            r"Datatype mismatch for overridden, FP16 \(local\) != FP32 \(remote\)",
        ),
    ],
)
def test_validate_model_tensor(
    model_tensor: OpenInferenceMetadataTensor,
    local_field: pydantic.fields.FieldInfo,
    is_input: bool,
    expected_error_message: str | None,
) -> None:
    with contextlib.ExitStack() as ctx:
        if expected_error_message is not None:
            ctx.enter_context(pytest.raises(IncompatibleTensorError, match=expected_error_message))
        validate_model_tensor(model_tensor, local_field, is_input=is_input)
