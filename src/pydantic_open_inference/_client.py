from __future__ import annotations

import atexit
import contextlib
import json
from typing import TYPE_CHECKING, cast

import asyncio_atexit
import httpx

from ._utils import (
    OpenInferenceAPIInput,
    OpenInferenceAPIOutput,
    OpenInferenceAPIRequestedOutput,
    OpenInferenceModelMetadata,
    PydanticOpenInferenceError,
    Singleton,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class OpenInferenceHTTPClientAPIError(PydanticOpenInferenceError):
    """Exceptions from the OpenInferenceHTTPClientAPI."""


class BadStatusCodeFromServerError(OpenInferenceHTTPClientAPIError):
    """A bad status code was returned from the open inference server."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        message = f"Inference server responded with HTTP {status_code}"
        super().__init__(message)

    @property
    def status_code(self) -> int:
        return self._status_code


class OpenInferenceHTTPClientAPI(metaclass=Singleton):
    def __init__(self, base_url: str) -> None:
        if not base_url.endswith("/"):
            base_url += "/"
        self._base_url = base_url
        self._headers = {"Content-Type": "application/json"}
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
            )
            atexit.register(self._sync_client.close)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
            )
            asyncio_atexit.register(self._async_client.aclose)
        return self._async_client

    @staticmethod
    def _create_infer_payload(
        inputs: list[OpenInferenceAPIInput], outputs: list[OpenInferenceAPIRequestedOutput] | None = None
    ) -> dict[str, list[OpenInferenceAPIInput] | list[OpenInferenceAPIRequestedOutput]]:
        payload: dict[str, list[OpenInferenceAPIInput] | list[OpenInferenceAPIRequestedOutput]] = {"inputs": inputs}
        if outputs is not None:
            payload["outputs"] = outputs
        return payload

    @staticmethod
    def _get_model_url(model_name: str, model_version: str | None) -> str:
        if model_version is None:
            return f"v2/models/{model_name}"
        return f"v2/models/{model_name}/versions/{model_version}"

    def _get_infer_url(self, model_name: str, model_version: str | None) -> str:
        return f"{self._get_model_url(model_name, model_version)}/infer"

    @contextlib.contextmanager
    def _handle_error(self) -> Generator[None]:
        try:
            yield
        except httpx.HTTPStatusError as error:
            raise BadStatusCodeFromServerError(
                status_code=error.response.status_code,
            ) from error
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise OpenInferenceHTTPClientAPIError(error) from error

    @staticmethod
    def _process_readiness_response(response: httpx.Response) -> bool:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            return False
        else:
            if not response.content:
                # No content indicates the "Predict Protocol - Version 2"
                # which is deceptively similar to the Open Inference V2 protocol
                # https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md
                # So the fact that the request was successful means the model is ready.
                return True
            # Content indicates the actual "Open Inference V2" protocol,
            # so the readiness is in the "Model Ready Response JSON Object"
            return cast("bool", response.json().get("ready", False))

    def model_metadata(
        self, model_name: str, model_version: str | None = None, timeout_seconds: float | None = None
    ) -> OpenInferenceModelMetadata:
        with self._handle_error():
            client = self._get_sync_client()
            response = client.get(
                self._get_model_url(model_name, model_version),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return cast("OpenInferenceModelMetadata", response.json())

    async def model_metadata_async(
        self, model_name: str, model_version: str | None = None, timeout_seconds: float | None = None
    ) -> OpenInferenceModelMetadata:
        with self._handle_error():
            client = self._get_async_client()
            response = await client.get(
                self._get_model_url(model_name, model_version),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return cast("OpenInferenceModelMetadata", response.json())

    def model_readiness(
        self, model_name: str, model_version: str | None = None, timeout_seconds: float | None = None
    ) -> bool:
        client = self._get_sync_client()
        response = client.get(
            f"{self._get_model_url(model_name, model_version)}/ready",
            timeout=timeout_seconds,
        )
        return self._process_readiness_response(response)

    async def model_readiness_async(
        self, model_name: str, model_version: str | None = None, timeout_seconds: float | None = None
    ) -> bool:
        client = self._get_async_client()
        response = await client.get(
            f"{self._get_model_url(model_name, model_version)}/ready",
            timeout=timeout_seconds,
        )
        return self._process_readiness_response(response)

    def infer(
        self,
        model_name: str,
        inputs: list[OpenInferenceAPIInput],
        outputs: list[OpenInferenceAPIRequestedOutput] | None = None,
        model_version: str | None = None,
        timeout_seconds: float | None = None,
    ) -> list[OpenInferenceAPIOutput]:
        with self._handle_error():
            client = self._get_sync_client()
            response = client.post(
                self._get_infer_url(model_name, model_version),
                json=self._create_infer_payload(inputs, outputs),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return cast("list[OpenInferenceAPIOutput]", response.json().get("outputs", []))

    async def infer_async(
        self,
        model_name: str,
        inputs: list[OpenInferenceAPIInput],
        outputs: list[OpenInferenceAPIRequestedOutput] | None = None,
        model_version: str | None = None,
        timeout_seconds: float | None = None,
    ) -> list[OpenInferenceAPIOutput]:
        with self._handle_error():
            client = self._get_async_client()
            response = await client.post(
                self._get_infer_url(model_name, model_version),
                json=self._create_infer_payload(inputs, outputs),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return cast("list[OpenInferenceAPIOutput]", response.json().get("outputs", []))
