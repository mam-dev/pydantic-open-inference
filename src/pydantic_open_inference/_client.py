from __future__ import annotations

import atexit
import json
from typing import cast

import httpx

from ._utils import (
    OpenInferenceAPIInput,
    OpenInferenceAPIOutput,
    OpenInferenceAPIRequestedOutput,
    PydanticOpenInferenceError,
    Singleton,
)


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
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Content-Type": "application/json"},
        )
        atexit.register(self._client.close)

    @staticmethod
    def _get_model_url(model_name: str, model_version: str | None) -> str:
        if model_version is None:
            return f"v2/models/{model_name}"
        return f"v2/models/{model_name}/versions/{model_version}"

    def model_readiness(
        self, model_name: str, model_version: str | None = None, timeout_seconds: float | None = None
    ) -> bool:
        try:
            response = self._client.get(
                f"{self._get_model_url(model_name, model_version)}/ready",
                timeout=timeout_seconds,
            ).raise_for_status()
        except httpx.HTTPStatusError:
            return False
        else:
            return cast("bool", response.json().get("ready", False))

    def infer(
        self,
        model_name: str,
        inputs: list[OpenInferenceAPIInput],
        outputs: list[OpenInferenceAPIRequestedOutput] | None = None,
        model_version: str | None = None,
        timeout_seconds: float | None = None,
    ) -> list[OpenInferenceAPIOutput]:
        try:
            payload: dict[str, list[OpenInferenceAPIInput] | list[OpenInferenceAPIRequestedOutput]] = {"inputs": inputs}
            if outputs is not None:
                payload["outputs"] = outputs
            response = self._client.post(
                f"{self._get_model_url(model_name, model_version)}/infer",
                json=payload,
                timeout=timeout_seconds,
            ).raise_for_status()
            return cast("list[OpenInferenceAPIOutput]", response.json().get("outputs", []))
        except httpx.HTTPStatusError as error:
            raise BadStatusCodeFromServerError(
                status_code=error.response.status_code,
            ) from error
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise OpenInferenceHTTPClientAPIError(error) from error
