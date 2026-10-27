"""Tests for the _client module."""

from __future__ import annotations

import atexit
import contextlib
from json import JSONDecodeError
from typing import Any
from unittest.mock import AsyncMock, Mock

import asyncio_atexit
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
    OpenInferenceModelMetadata,
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


@pytest.fixture(autouse=True)
def mock_httpx_async_client_cls(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        "pydantic_open_inference._client.httpx.AsyncClient",
        mock_client_cls := Mock(spec=httpx.AsyncClient),
    )
    mock_client_cls.return_value.post = AsyncMock(return_value=Mock())
    return mock_client_cls


@pytest.mark.parametrize("with_trailing_slash", [True, False])
def test_client_api_instantiation_and_lazy_clients(
    monkeypatch: pytest.MonkeyPatch,
    mock_httpx_client_cls: Mock,
    mock_httpx_async_client_cls: Mock,
    with_trailing_slash: bool,
) -> None:
    monkeypatch.setattr(
        "pydantic_open_inference._client.atexit.register",
        mock_atexit_register := Mock(spec=atexit.register),
    )
    monkeypatch.setattr(
        "pydantic_open_inference._client.asyncio_atexit.register",
        mock_asyncio_atexit_register := Mock(spec=asyncio_atexit.register),
    )
    base_url = "https://server"
    if with_trailing_slash:
        base_url += "/"
    client = OpenInferenceHTTPClientAPI(base_url=base_url)

    mock_httpx_client_cls.assert_not_called()
    mock_httpx_async_client_cls.assert_not_called()
    mock_atexit_register.assert_not_called()
    mock_asyncio_atexit_register.assert_not_called()

    for _ in range(2):
        _ = client._get_sync_client()

        mock_httpx_client_cls.assert_called_once_with(
            base_url="https://server/",
            headers={"Content-Type": "application/json"},
        )
        mock_atexit_register.assert_called_once_with(mock_httpx_client_cls.return_value.close)

    for _ in range(2):
        _ = client._get_async_client()

        mock_httpx_async_client_cls.assert_called_once_with(
            base_url="https://server/",
            headers={"Content-Type": "application/json"},
        )
        mock_asyncio_atexit_register.assert_called_once_with(mock_httpx_async_client_cls.return_value.aclose)


def test_client_api_is_singleton() -> None:
    assert isinstance(OpenInferenceHTTPClientAPI, Singleton)


@pytest.mark.parametrize(
    "model_version, response, timeout, expected",
    [
        (
            None,
            httpx.Response(
                status_code=200,
                json=(metadata := {"name": "my_model", "platform": "triton", "inputs": [], "outputs": []}),
            ),
            None,
            metadata,
        ),
        (
            "1.0",
            httpx.Response(
                status_code=200,
                json=(
                    metadata := {
                        "name": "my_model",
                        "versions": ["1.0"],
                        "platform": "triton",
                        "inputs": [],
                        "outputs": [],
                    }
                ),
            ),
            None,
            metadata,
        ),
        (
            None,
            httpx.Response(status_code=500),
            None,
            BadStatusCodeFromServerError(status_code=500),
        ),
    ],
)
def test_client_api_model_metadata(
    mock_httpx_client_cls: Mock,
    model_version: str | None,
    response: httpx.Response,
    timeout: float | None,
    expected: OpenInferenceModelMetadata | BadStatusCodeFromServerError,
) -> None:
    response.request = Mock(spec=httpx.Request, url="http://some/url")
    mock_httpx_client_cls.return_value.get.return_value = response
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    with contextlib.ExitStack() as ctx:
        if isinstance(expected, BadStatusCodeFromServerError):
            ctx.enter_context(pytest.raises(BadStatusCodeFromServerError, match=rf"{response.status_code}"))
        actual = api.model_metadata(
            "my_model",
            model_version=model_version,
            timeout_seconds=timeout,
        )
    if not isinstance(expected, BadStatusCodeFromServerError):
        assert actual == expected
    mock_httpx_client_cls.return_value.get.assert_called_once_with(
        "v2/models/my_model" if model_version is None else f"v2/models/my_model/versions/{model_version}",
        timeout=timeout,
    )


@pytest.mark.parametrize(
    "model_version, response, timeout, expected",
    [
        (
            None,
            httpx.Response(status_code=200, json={"name": "my_model", "ready": True}),
            None,
            True,
        ),
        (
            "1.0",
            httpx.Response(status_code=200, json={"name": "my_model", "ready": True}),
            5.0,
            True,
        ),
        (
            None,
            httpx.Response(status_code=200, json={"name": "my_model", "ready": False}),
            None,
            False,
        ),
        (
            None,
            httpx.Response(status_code=200),
            None,
            True,
        ),
        (
            "1.0",
            httpx.Response(status_code=200),
            5.0,
            True,
        ),
        (
            None,
            httpx.Response(status_code=400),
            None,
            False,
        ),
        (
            None,
            httpx.Response(status_code=404),
            None,
            False,
        ),
        (
            None,
            httpx.Response(status_code=408),
            None,
            False,
        ),
    ],
)
def test_client_api_model_readiness(
    mock_httpx_client_cls: Mock,
    model_version: str | None,
    response: httpx.Response,
    timeout: float | None,
    expected: bool,
) -> None:
    response.request = Mock(spec=httpx.Request, url="http://some/url")
    mock_httpx_client_cls.return_value.get.return_value = response
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    actual = api.model_readiness(
        "my_model",
        model_version=model_version,
        timeout_seconds=timeout,
    )
    assert actual == expected
    mock_httpx_client_cls.return_value.get.assert_called_once_with(
        "v2/models/my_model/ready" if model_version is None else f"v2/models/my_model/versions/{model_version}/ready",
        timeout=timeout,
    )


@pytest.mark.parametrize("with_version", [pytest.param(False, id="no-version"), pytest.param(True, id="with-version")])
@pytest.mark.parametrize(
    "inputs, outputs, timeout, expected_payload",
    INFER_PARAMETERS := [
        pytest.param([], None, None, {"inputs": []}, id="empty"),
        pytest.param(
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
            id="single-text-input",
        ),
        pytest.param(
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
            id="single-text-input-with-output",
        ),
    ],
)
def test_client_api_infer(
    mock_httpx_client_cls: Mock,
    inputs: list[OpenInferenceAPIInput],
    outputs: list[OpenInferenceAPIRequestedOutput] | None,
    timeout: float | None,
    expected_payload: dict[str, Any],
    with_version: bool,
) -> None:
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    extra_kwargs: dict[str, Any] = {}
    if with_version:
        extra_kwargs["model_version"] = "1.2.3"
    actual = api.infer(
        "my_model",
        inputs=inputs,
        outputs=outputs,
        timeout_seconds=timeout,
        **extra_kwargs,
    )
    mock_httpx_client_cls.return_value.post.assert_called_once_with(
        "v2/models/my_model/versions/1.2.3/infer" if with_version else "v2/models/my_model/infer",
        json=expected_payload,
        timeout=timeout,
    )
    mock_response = mock_httpx_client_cls.return_value.post.return_value
    mock_response.raise_for_status.assert_called_once_with()
    mock_response.json.return_value.get.assert_called_once_with("outputs", [])
    mock_json_data = mock_response.json.return_value.get.return_value
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
    mock_response.json.side_effect = JSONDecodeError("fake", "doc", 0)
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


@pytest.mark.parametrize("with_version", [pytest.param(False, id="no-version"), pytest.param(True, id="with-version")])
@pytest.mark.parametrize(
    "inputs, outputs, timeout, expected_payload",
    INFER_PARAMETERS,
)
@pytest.mark.asyncio
async def test_client_api_infer_async(
    mock_httpx_async_client_cls: Mock,
    inputs: list[OpenInferenceAPIInput],
    outputs: list[OpenInferenceAPIRequestedOutput] | None,
    timeout: float | None,
    expected_payload: dict[str, Any],
    with_version: bool,
) -> None:
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    extra_kwargs: dict[str, Any] = {}
    if with_version:
        extra_kwargs["model_version"] = "1.2.3"
    actual = await api.infer_async(
        "my_model",
        inputs=inputs,
        outputs=outputs,
        timeout_seconds=timeout,
        **extra_kwargs,
    )
    mock_httpx_async_client_cls.return_value.post.assert_awaited_once_with(
        "v2/models/my_model/versions/1.2.3/infer" if with_version else "v2/models/my_model/infer",
        json=expected_payload,
        timeout=timeout,
    )
    mock_response = mock_httpx_async_client_cls.return_value.post.return_value
    mock_response.raise_for_status.assert_called_once_with()
    mock_response.json.return_value.get.assert_called_once_with("outputs", [])
    mock_json_data = mock_response.json.return_value.get.return_value
    assert actual is mock_json_data


@pytest.mark.asyncio
async def test_client_api_infer_async_http_error(
    mock_httpx_async_client_cls: Mock,
) -> None:
    status_code = 429
    mock_response = mock_httpx_async_client_cls.return_value.post.return_value
    mock_response.raise_for_status.side_effect = HTTPStatusError(
        "fake", request=Mock(), response=Mock(status_code=status_code)
    )
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    with pytest.raises(
        BadStatusCodeFromServerError,
        match=f"HTTP {status_code}",
        check=lambda error: error.status_code == status_code,
    ):
        _ = await api.infer_async(
            "my_model",
            inputs=[],
        )
    mock_response.raise_for_status.assert_called_once_with()
    mock_httpx_async_client_cls.return_value.post.assert_awaited_once_with(
        "v2/models/my_model/infer",
        json={"inputs": []},
        timeout=None,
    )


@pytest.mark.asyncio
async def test_client_api_infer_async_json_error(
    mock_httpx_async_client_cls: Mock,
) -> None:
    mock_response = mock_httpx_async_client_cls.return_value.post.return_value
    mock_response.json.side_effect = JSONDecodeError("fake", "doc", 0)
    api = OpenInferenceHTTPClientAPI(base_url="https://server/")
    with pytest.raises(OpenInferenceHTTPClientAPIError):
        _ = await api.infer_async(
            "my_model",
            inputs=[],
        )
    mock_response.raise_for_status.assert_called_once_with()
    mock_httpx_async_client_cls.return_value.post.assert_awaited_once_with(
        "v2/models/my_model/infer",
        json={"inputs": []},
        timeout=None,
    )
