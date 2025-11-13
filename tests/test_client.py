"""Tests for the _client module."""

from __future__ import annotations

import atexit
from json import JSONDecodeError
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from httpx import HTTPStatusError

from pydantic_open_inference._client import (
    BadStatusCodeFromServerError,
    OpenInferenceHTTPClientAPI,
    OpenInferenceHTTPClientAPIError,
)
from pydantic_open_inference._utils import (
    OpenInferenceAPIInput,
    OpenInferenceAPIRequestedOutput,
    Singleton,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    Singleton._instances.clear()


@pytest.fixture(autouse=True)
def mock_httpx_client_cls(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "pydantic_open_inference._client.httpx.Client",
        mock_client_cls := Mock(spec=httpx.Client),
    )
    return mock_client_cls


@pytest.mark.parametrize("with_trailing_slash", [True, False])
def test_client_api_instantiation(
    monkeypatch: pytest.MonkeyPatch,
    mock_httpx_client_cls: Mock,
    with_trailing_slash: bool,
) -> None:
    monkeypatch.setattr(
        "pydantic_open_inference._client.atexit.register",
        mock_atexit_register := Mock(spec=atexit.register),
    )
    base_url = "https://server"
    if with_trailing_slash:
        base_url += "/"
    _ = OpenInferenceHTTPClientAPI(base_url=base_url)
    mock_httpx_client_cls.assert_called_once_with(
        base_url="https://server/",
        headers={"Content-Type": "application/json"},
    )
    mock_atexit_register.assert_called_once_with(mock_httpx_client_cls.return_value.close)


def test_client_api_is_singleton() -> None:
    assert isinstance(OpenInferenceHTTPClientAPI, Singleton)


@pytest.mark.parametrize(
    "inputs, outputs, timeout, expected_payload",
    INFER_PARAMETERS := [
        ([], None, None, {"inputs": []}),
        (
            [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello world"],
                }
            ],
            None,
            5.0,
            {
                "inputs": [
                    {
                        "name": "text",
                        "datatype": "BYTES",
                        "shape": [1],
                        "data": ["hello world"],
                    }
                ]
            },
        ),
        (
            [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello world"],
                }
            ],
            [{"name": "answer"}],
            3.5,
            {
                "inputs": [
                    {
                        "name": "text",
                        "datatype": "BYTES",
                        "shape": [1],
                        "data": ["hello world"],
                    }
                ],
                "outputs": [{"name": "answer"}],
            },
        ),
    ],
)
def test_client_api_infer(
    mock_httpx_client_cls: Mock,
    inputs: list[OpenInferenceAPIInput],
    outputs: list[OpenInferenceAPIRequestedOutput] | None,
    timeout: float | None,
    expected_payload: dict[str, Any],
) -> None:
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    actual = api.infer(
        "my_model",
        inputs=inputs,
        outputs=outputs,
        timeout_seconds=timeout,
    )
    mock_httpx_client_cls.return_value.post.assert_called_once_with(
        "v2/models/my_model/infer",
        json=expected_payload,
        timeout=timeout,
    )
    mock_response = mock_httpx_client_cls.return_value.post.return_value
    mock_response.raise_for_status.return_value.json.return_value.get.assert_called_once_with("outputs", [])
    assert actual is mock_response.raise_for_status.return_value.json.return_value.get.return_value


@pytest.mark.parametrize("inputs, outputs, timeout, expected_payload", INFER_PARAMETERS)
def test_client_api_infer__with_version(
    mock_httpx_client_cls: Mock,
    inputs: list[OpenInferenceAPIInput],
    outputs: list[OpenInferenceAPIRequestedOutput] | None,
    timeout: float | None,
    expected_payload: dict[str, Any],
) -> None:
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    actual = api.infer(
        model_name="my_model",
        model_version="1.2.3",
        inputs=inputs,
        outputs=outputs,
        timeout_seconds=timeout,
    )
    mock_httpx_client_cls.return_value.post.assert_called_once_with(
        "v2/models/my_model/versions/1.2.3/infer",
        json=expected_payload,
        timeout=timeout,
    )
    mock_response = mock_httpx_client_cls.return_value.post.return_value
    mock_response.raise_for_status.return_value.json.return_value.get.assert_called_once_with("outputs", [])
    mock_json_data = mock_response.raise_for_status.return_value.json.return_value.get.return_value
    assert actual is mock_json_data


def test_client_api_infer_http_error(
    mock_httpx_client_cls: Mock,
) -> None:
    status_code = 429
    mock_response = mock_httpx_client_cls.return_value.post.return_value
    mock_response.raise_for_status.side_effect = HTTPStatusError(
        "fake", request=Mock(), response=Mock(status_code=status_code)
    )
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    with pytest.raises(
        BadStatusCodeFromServerError,
        match=f"HTTP {status_code}",
        check=lambda error: error.status_code == status_code,
    ):
        _ = api.infer(
            "my_model",
            inputs=[],
        )

    mock_httpx_client_cls.return_value.post.assert_called_once_with(
        "v2/models/my_model/infer",
        json={"inputs": []},
        timeout=None,
    )


def test_client_api_infer_json_error(
    mock_httpx_client_cls: Mock,
) -> None:
    mock_response = mock_httpx_client_cls.return_value.post.return_value
    mock_response.raise_for_status.return_value.json.side_effect = JSONDecodeError("fake", "doc", 0)
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    with pytest.raises(OpenInferenceHTTPClientAPIError):
        _ = api.infer(
            "my_model",
            inputs=[],
        )
    mock_httpx_client_cls.return_value.post.assert_called_once_with(
        "v2/models/my_model/infer",
        json={"inputs": []},
        timeout=None,
    )
