"""Tests for the _remote_model module."""

from __future__ import annotations

import itertools
from collections import namedtuple
from typing import Annotated, TypeVar
from unittest.mock import Mock

import pytest

from pydantic_open_inference._client import OpenInferenceHTTPClientAPI
from pydantic_open_inference._remote_model import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)
from pydantic_open_inference._utils import (
    DatatypeOverride,
    OpenInferenceAPIOutput,
    OpenInferenceAPIRequestedOutput,
)


class IntTuplesOutputsBaseModel(OutputsBaseModel):
    values: list[tuple[int, int]]


Point = namedtuple("Point", ["x", "y"])


class NamedTuplesOutputsBaseModel(OutputsBaseModel):
    values: list[Point]


class SingleStringOutputsBaseModel(OutputsBaseModel):
    text: str


OutputsModelT = TypeVar("OutputsModelT", bound=OutputsBaseModel)


@pytest.mark.parametrize(
    "model_cls, outputs, expected",
    [
        pytest.param(
            IntTuplesOutputsBaseModel,
            [{"name": "values", "shape": [0], "data": []}],
            IntTuplesOutputsBaseModel(values=[]),
            id="empty",
        ),
        pytest.param(
            IntTuplesOutputsBaseModel,
            [{"name": "values", "shape": [2, 2], "data": [[1, 2], [3, 4]]}],
            IntTuplesOutputsBaseModel(values=[(1, 2), (3, 4)]),
            id="simple",
        ),
        pytest.param(
            IntTuplesOutputsBaseModel,
            [{"name": "values", "shape": [2, 2], "data": [["1", "2"], ["3", "4"]]}],
            IntTuplesOutputsBaseModel(values=[(1, 2), (3, 4)]),
            id="str-to-int",
        ),
        pytest.param(
            NamedTuplesOutputsBaseModel,
            [{"name": "values", "shape": [2, 2], "data": [[1, 2], [3, 4]]}],
            NamedTuplesOutputsBaseModel(values=[Point(x=1, y=2), Point(x=3, y=4)]),
            id="namedtuple",
        ),
        pytest.param(
            SingleStringOutputsBaseModel,
            [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello world"],
                }
            ],
            SingleStringOutputsBaseModel(text="hello world"),
            id="string",
        ),
    ],
)
def test_outputs_model_from_outputs(
    model_cls: type[OutputsModelT],
    outputs: list[OpenInferenceAPIOutput],
    expected: OutputsModelT,
) -> None:
    assert model_cls.from_outputs(outputs) == expected


@pytest.mark.parametrize(
    "model_cls, expected",
    [
        (IntTuplesOutputsBaseModel, [{"name": "values"}]),
        (NamedTuplesOutputsBaseModel, [{"name": "values"}]),
    ],
)
def test_outputs_model_get_requested_outputs(
    model_cls: type[OutputsBaseModel], expected: list[OpenInferenceAPIRequestedOutput]
) -> None:
    assert model_cls.get_requested_outputs() == expected


class IntTuplesInputsBaseModel(InputsBaseModel):
    values: list[tuple[int, int]]


class NamedTuplesInputsBaseModel(InputsBaseModel):
    values: list[Point]


class SingleStringInputsBaseModel(InputsBaseModel):
    text: str


class Int16TuplesInputsBaseModel(InputsBaseModel):
    values: Annotated[list[tuple[int, int]], DatatypeOverride("INT16")]


@pytest.mark.parametrize(
    "model_instance, expected",
    [
        pytest.param(
            IntTuplesInputsBaseModel(values=[(1, 2), (3, 4)]),
            [
                {
                    "name": "values",
                    "datatype": "INT64",
                    "shape": [2, 2],
                    "data": [[1, 2], [3, 4]],
                }
            ],
            id="2x2-int",
        ),
        pytest.param(
            NamedTuplesInputsBaseModel(values=[Point(x=1, y=2), Point(x=3, y=4)]),
            [
                {
                    "name": "values",
                    "datatype": "INT64",
                    "shape": [2, 2],
                    "data": [[1, 2], [3, 4]],
                }
            ],
            id="namedtuple",
        ),
        pytest.param(
            SingleStringInputsBaseModel(text="hello world"),
            [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello world"],
                }
            ],
            id="string",
        ),
        pytest.param(
            IntTuplesInputsBaseModel(values=[(1, 2)]),
            [
                {
                    "name": "values",
                    "datatype": "INT64",
                    "shape": [1, 2],
                    "data": [[1, 2]],
                }
            ],
            id="1x2",
        ),
        pytest.param(
            Int16TuplesInputsBaseModel(values=[(1, 2)]),
            [
                {
                    "name": "values",
                    "datatype": "INT16",
                    "shape": [1, 2],
                    "data": [[1, 2]],
                }
            ],
            id="override",
        ),
    ],
)
def test_inputs_model_to_inputs(
    model_instance: InputsBaseModel,
    expected: list[OpenInferenceAPIOutput],
) -> None:
    assert model_instance.to_inputs() == expected


@pytest.fixture
def mock_client_api_cls(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "pydantic_open_inference._remote_model.OpenInferenceHTTPClientAPI",
        mock_api_cls := Mock(spec=OpenInferenceHTTPClientAPI),
    )
    return mock_api_cls


def test_remote_model_instantiate(mock_client_api_cls: Mock) -> None:
    _ = RemoteModel(
        model_name="my_model",
        inputs_model=IntTuplesInputsBaseModel,
        outputs_model=IntTuplesOutputsBaseModel,
        server_url="https://server/",
    )
    mock_client_api_cls.assert_called_once_with(
        base_url="https://server/",
    )


@pytest.mark.parametrize("with_version, timeout", itertools.product([False, True], [None, 4.5]))
def test_remote_model_infer(mock_client_api_cls: Mock, with_version: bool, timeout: float | None) -> None:
    fake_model_name = "my_model"
    fake_version = "1.2.3"
    mock_outputs_model = Mock(spec=OutputsBaseModel)
    remote_model: RemoteModel[IntTuplesInputsBaseModel, OutputsBaseModel] = RemoteModel(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        inputs_model=IntTuplesInputsBaseModel,
        outputs_model=mock_outputs_model,
        server_url="https://server/",
        request_timeout_seconds=timeout,
    )
    actual = remote_model.infer(mock_inputs := Mock(spec=IntTuplesInputsBaseModel))
    mock_client_api_cls.return_value.infer.assert_called_once_with(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        inputs=mock_inputs.to_inputs.return_value,
        outputs=mock_outputs_model.get_requested_outputs.return_value,
        timeout_seconds=timeout,
    )
    assert actual == mock_outputs_model.from_outputs.return_value
    mock_outputs_model.from_outputs.assert_called_once_with(mock_client_api_cls.return_value.infer.return_value)


def test_remote_model_infer__wrong_type(mock_client_api_cls: Mock) -> None:
    mock_outputs_model = Mock(spec=OutputsBaseModel)
    remote_model: RemoteModel[IntTuplesInputsBaseModel, OutputsBaseModel] = RemoteModel(
        model_name="my_model",
        inputs_model=IntTuplesInputsBaseModel,
        outputs_model=mock_outputs_model,
        server_url="https://server/",
    )
    bad_input = NamedTuplesInputsBaseModel(values=[Point(x=1, y=2)])
    with pytest.raises(TypeError):
        _ = remote_model.infer(bad_input)  # type: ignore[arg-type]
    mock_client_api_cls.return_value.infer.assert_not_called()
