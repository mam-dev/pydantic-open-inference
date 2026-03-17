"""Tests for the _remote_model module."""

from __future__ import annotations

import itertools
from collections import namedtuple
from typing import Annotated, TypeVar
from unittest.mock import Mock, call

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
    get_input_tensor_by_name,
    get_output_tensor_by_name,
    validate_model_tensor,
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


@pytest.mark.parametrize("with_version, timeout", itertools.product([True, False], [None, 3.4]))
def test_remote_model_validate(
    monkeypatch: pytest.MonkeyPatch, mock_client_api_cls: Mock, with_version: bool, timeout: float | None
) -> None:
    fake_model_name = "my_model"
    fake_version = "1.2.3"
    monkeypatch.setattr(
        "pydantic_open_inference._remote_model.get_input_tensor_by_name",
        mock_get_input_tensor := Mock(spec=get_input_tensor_by_name),
    )
    monkeypatch.setattr(
        "pydantic_open_inference._remote_model.get_output_tensor_by_name",
        mock_get_output_tensor := Mock(spec=get_output_tensor_by_name),
    )
    monkeypatch.setattr(
        "pydantic_open_inference._remote_model.validate_model_tensor",
        mock_validate_model_tensor := Mock(spec=validate_model_tensor),
    )
    remote_model: RemoteModel[IntTuplesInputsBaseModel, OutputsBaseModel] = RemoteModel(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        inputs_model=IntTuplesInputsBaseModel,
        outputs_model=IntTuplesOutputsBaseModel,
        server_url="https://server/",
        request_timeout_seconds=timeout,
    )

    remote_model.validate()

    mock_client_api_cls.return_value.model_metadata.assert_called_once_with(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        timeout_seconds=timeout,
    )
    mock_get_input_tensor.assert_called_once_with(
        "values", mock_client_api_cls.return_value.model_metadata.return_value
    )
    mock_get_output_tensor.assert_called_once_with(
        "values", mock_client_api_cls.return_value.model_metadata.return_value
    )
    assert mock_validate_model_tensor.mock_calls == [
        call(mock_get_input_tensor.return_value, IntTuplesInputsBaseModel.model_fields["values"], is_input=True),
        call(mock_get_output_tensor.return_value, IntTuplesOutputsBaseModel.model_fields["values"], is_input=False),
    ]


@pytest.mark.parametrize("with_version, timeout", itertools.product([True, False], [None, 3.4]))
def test_remote_model_is_ready(mock_client_api_cls: Mock, with_version: bool, timeout: float | None) -> None:
    fake_model_name = "my_model"
    fake_version = "1.2.3"
    remote_model: RemoteModel[IntTuplesInputsBaseModel, OutputsBaseModel] = RemoteModel(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        inputs_model=IntTuplesInputsBaseModel,
        outputs_model=Mock(spec=OutputsBaseModel),
        server_url="https://server/",
        request_timeout_seconds=timeout,
    )
    actual = remote_model.is_ready()
    assert actual == mock_client_api_cls.return_value.model_readiness.return_value
    mock_client_api_cls.return_value.model_readiness.assert_called_once_with(
        model_name=fake_model_name,
        model_version=fake_version if with_version else None,
        timeout_seconds=timeout,
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
