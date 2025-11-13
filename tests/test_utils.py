"""Tests for the _utils module."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic.fields import FieldInfo

from pydantic_open_inference._utils import (
    Data,
    Datatype,
    DatatypeOverride,
    Shape,
    ShapeDataMismatchError,
    Singleton,
    get_data,
    get_datatype,
    get_shape,
    is_flat,
    is_listlike,
    parse_row_major_order,
    unflatten_data,
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
