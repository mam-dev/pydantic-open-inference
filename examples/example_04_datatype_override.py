"""Example: Use DatatypeOverride to control the model input datatypes."""

from typing import Annotated

from pydantic_open_inference import (
    DatatypeOverride,
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)


class MatrixMultiplicationInput(InputsBaseModel):
    """Input matrices for multiplication.

    Using INT32 instead of the default INT64 to match model requirements.
    """

    # Matrix A: 2x3 matrix of integers
    matrix_a: Annotated[tuple[tuple[int, int, int], tuple[int, int, int]], DatatypeOverride("INT32")]
    # Matrix B: 3x2 matrix of integers
    matrix_b: Annotated[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], DatatypeOverride("INT32")]


class MatrixMultiplicationOutput(OutputsBaseModel):
    """Result of matrix multiplication."""

    # Result: 2x2 matrix
    result: tuple[tuple[int, int], tuple[int, int]]


matrix_model = RemoteModel(
    model_name="matrix_multiply",
    inputs_model=MatrixMultiplicationInput,
    outputs_model=MatrixMultiplicationOutput,
    server_url="http://localhost:8000",
)


matrix_input = MatrixMultiplicationInput(
    matrix_a=(
        (1, 2, 3),
        (4, 5, 6),
    ),
    matrix_b=(
        (7, 8),
        (9, 10),
        (11, 12),
    ),
)

matrix_result = matrix_model.infer(matrix_input)
print(f"Matrix multiplication result: {matrix_result.result}")
