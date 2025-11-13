"""Tests for the examples directory."""

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from pydantic_open_inference import (
    BadStatusCodeFromServerError,
    OutputsBaseModel,
    PydanticOpenInferenceError,
    RemoteModel,
)
from pydantic_open_inference._utils import (
    OpenInferenceAPIInput,
    OpenInferenceAPIOutput,
    OpenInferenceAPIRequestedOutput,
)

EXAMPLES = [
    EXAMPLE_01A := "example_01a_text_cls",
    EXAMPLE_01B := "example_01b_text_cls",
    EXAMPLE_02 := "example_02_batched_img_embedding",
    EXAMPLE_03 := "example_03_ner_named_tuples",
    EXAMPLE_04 := "example_04_datatype_override",
    EXAMPLE_05 := "example_05_multioutput",
    EXAMPLE_06 := "example_06_error_handling",
]


def test_examples_listed() -> None:
    found_examples = {p.stem for p in (Path(__file__).parent.parent / "examples").glob("*.py")}
    listed_examples = set(EXAMPLES)
    assert listed_examples == found_examples


def _run_example(example: str) -> None:
    example_module_name = f"examples.{example}"
    try:
        module = sys.modules[example_module_name]
        importlib.reload(module)
    except KeyError:
        _ = importlib.import_module(example_module_name)


class MockRemoteModelSideEffect:
    """A 'side effect' for mocks of RemoteModel."""

    def __init__(self, raw_output_model: dict[str, Any]) -> None:
        """Instantiate a side effect.

        Args:
            raw_output_model: A dict to use to instantiate
                the model-output pydantic-model.

        """
        self._raw_output_model = raw_output_model

    def __call__(self, **kwargs: Any) -> Mock:
        """The actual side effect call.

        The signature is the same as RemoteModel init.

        """
        outputs_model_cls = kwargs.pop("outputs_model")
        assert isinstance(outputs_model_cls, type)
        assert issubclass(outputs_model_cls, OutputsBaseModel)

        mock_outputs = outputs_model_cls.model_validate(self._raw_output_model)
        return Mock(spec=RemoteModel, infer=Mock(spec=RemoteModel.infer, return_value=mock_outputs))


@pytest.mark.parametrize(
    "example, raw_outputs, expected_stdout",
    [
        (EXAMPLE_01A, raw_01_output := {"label": "spam", "confidence": 0.85}, stdout_01 := ["spam", "0.85"]),
        (EXAMPLE_01B, raw_01_output, stdout_01),
        (
            EXAMPLE_02,
            {"embeddings": [[0.3] * 512] * 4},
            ["Generated 4 embeddings", "Each embedding has dimension: 512"],
        ),
        (
            EXAMPLE_03,
            {"entities": [["This is a text", "label", 0.8]]},
            ["Entity: 'This is a text'", "label", "Score: 0.800"],
        ),
        (EXAMPLE_04, {"result": ((58, 64), (139, 154))}, ["Matrix multiplication result: ((58, 64), (139, 154))"]),
        (
            EXAMPLE_05,
            {"classification": "B", "probability": 0.7, "attention_weights": [0.1, 0.2, 0.3]},
            [" B", "0.7", "[0.1, 0.2, 0.3]"],
        ),
        (EXAMPLE_06, raw_01_output, stdout_01),
    ],
)
def test_run_example(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example: str,
    raw_outputs: dict[str, Any],
    expected_stdout: list[str],
) -> None:
    monkeypatch.setattr(
        "pydantic_open_inference.RemoteModel",
        mock_remote_model_cls := Mock(spec=RemoteModel, side_effect=MockRemoteModelSideEffect(raw_outputs)),
    )

    _run_example(example)

    mock_remote_model_cls.assert_called()
    stdout, stderr = capsys.readouterr()
    assert not stderr
    for expected_str in expected_stdout:
        assert expected_str in stdout


@pytest.mark.parametrize(
    "error, expected_stdout",
    [
        (PydanticOpenInferenceError("fake"), ["Inference failed", "fake"]),
        (BadStatusCodeFromServerError(status_code=429), ["HTTP429"]),
        (TypeError, ["Input type mismatch"]),
    ],
)
def test_run_error_handling_example(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception, expected_stdout: list[str]
) -> None:
    monkeypatch.setattr(
        "pydantic_open_inference.RemoteModel",
        mock_remote_model_cls := Mock(
            spec=RemoteModel, return_value=Mock(spec=RemoteModel, infer=Mock(spe=RemoteModel.infer, side_effect=error))
        ),
    )

    _run_example(EXAMPLE_06)

    mock_remote_model_cls.assert_called()
    stdout, stderr = capsys.readouterr()
    assert not stderr
    for expected_str in expected_stdout:
        assert expected_str in stdout


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "example, expected_url, expected_timeout, expected_request, outputs, expected_stdout",
    [
        (
            EXAMPLE_01A,
            "v2/models/text_classifier/infer",
            10.0,
            inputs_01 := {
                "inputs": [{"name": "text", "shape": [1], "datatype": "BYTES", "data": ["This is a great product!"]}],
                "outputs": [{"name": "label"}, {"name": "confidence"}],
            },
            outputs_01 := [
                {"name": "label", "datatype": "BYTES", "shape": [1], "data": ["spam"]},
                {"name": "confidence", "datatype": "FP32", "shape": [1], "data": ["0.85"]},
            ],
            stdout_01,
        ),
        (
            EXAMPLE_01B,
            "v2/models/text_classifier/infer",
            10.0,
            inputs_01,
            outputs_01,
            stdout_01,
        ),
        (
            EXAMPLE_02,
            "v2/models/resnet50_embeddings/versions/1/infer",
            None,
            {
                "inputs": [
                    {
                        "name": "images",
                        "datatype": "FP32",
                        "shape": [4, 150528],
                        "data": [
                            [0.5] * 150528,  # Image 1 (dummy flattened pixel values)
                            [0.3] * 150528,  # Image 2
                            [0.7] * 150528,  # Image 3
                            [0.2] * 150528,  # Image 4
                        ],
                    },
                ],
                "outputs": [{"name": "embeddings"}],
            },
            [
                {
                    "name": "embeddings",
                    "data": [[0.3] * 512] * 4,
                    "shape": [4, 512],
                    "datatype": "FP32",
                }
            ],
            [],
        ),
    ],
)
def test_run_example__integration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example: str,
    expected_url: str,
    expected_timeout: float | None,
    expected_request: dict[str, list[OpenInferenceAPIInput] | list[OpenInferenceAPIRequestedOutput]],
    outputs: list[OpenInferenceAPIOutput],
    expected_stdout: list[str],
) -> None:
    mock_post = Mock(spec=httpx.Client.post)
    mock_response = mock_post.return_value
    mock_response.raise_for_status.return_value = mock_response
    mock_response.json.return_value = {"outputs": outputs}
    monkeypatch.setattr("pydantic_open_inference._client.httpx.Client.post", mock_post)

    _run_example(example)

    assert mock_post.mock_calls[0].args == (expected_url,)
    assert mock_post.mock_calls[0].kwargs["json"] == expected_request
    assert mock_post.mock_calls[0].kwargs["timeout"] == expected_timeout
    mock_response.json.assert_called()
    stdout, stderr = capsys.readouterr()
    assert not stderr
    for expected_str in expected_stdout:
        assert expected_str in stdout
