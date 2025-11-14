# pydantic-open-inference

[![image](https://img.shields.io/pypi/v/pydantic-open-inference
)](https://pypi.python.org/pypi/pydantic-open-inference)
[![image](https://img.shields.io/github/license/mam-dev/pydantic-open-inference
)](https://github.com/mam-dev/pydantic-open-inference/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/pydantic-open-inference)](https://pypi.python.org/pypi/pydantic-open-inference)
[![Actions status](https://github.com/mam-dev/pydantic-open-inference/actions/workflows/ci.yaml/badge.svg)](https://github.com/mam-dev/pydantic-open-inference/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://www.mypy-lang.org)

## What is it?
Provides a client-side python interface for calling [open inference servers](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol).
Define [pydantic](https://docs.pydantic.dev/latest/) models for the model inputs and outputs, and it gets automatically
converted to and from the open inference protocol, removing the need for boilerplate
coding handling raw tensors and their shapes.

## Features
* ✅ Simple, type-safe API toward an open inference server using user-defined pydantic models
* ✅ Automatic shape inference from pydantic models
* ✅ Automatic datatype mapping (`int`→INT64, `float`→FP32, `str`→BYTES)
* ✅ Datatype override for fine-grained control of model inputs
* ✅ Automatically request only a subset of model outputs by omitting fields in your pydantic model
* ✅ Take advantage of the full power of pydantic to validate and type-coerce your model calls
* ✅ Multi-input/output models - each input/output maps to a pydantic field
* ✅ Custom timeout configuration - configurable per model
* ✅ Support calling inference for versioned and unversioned models
* ✅ Error handling with custom exceptions - handle, e.g., specific error codes from the inference server differently
* ✅ Fully typed - use, e.g., [mypy](https://mypy-lang.org/) and its [pydantic plugin](https://docs.pydantic.dev/latest/integrations/mypy/) to type check your code

## Installation
Installation using [pip](https://github.com/pypa/pip):
```bash
pip install pydantic-open-inference
```

Adding it to your project dependencies using [uv](https://docs.astral.sh/uv/):
```bash
uv add pydantic-open-inference
```

## How to use it
Assume we have an inference server serving an Open Inference V2 REST API at `http://localhost:8080`
and that the server contains a model named `example_model`, which has the following model **inputs**

| name | shape | datatype | description              | example data       |
|------|-------|----------|--------------------------|--------------------|
| text | [1]   | BYTES    | A single string of text. | `["This is a text"]` |

and the following model **outputs**:

| name     | shape   | datatype | description                                                                                                                                                                                                                                  | example data                                                     |
|----------|---------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| entities | [-1, 4] | BYTES    | <div style="width:200px">A list of entities extracted from the text. Each entity is made up of (in order) a label (either "order-id" or "invoice-id"), a confidence score, a start index for the entity in the text, and an end index.</div> | `[["order-id", "0.9", "23", "30"],["invoice-id","0.8","34","46"]]` |

We then define a pydantic model for the inputs, inheriting from `InputsBaseModel`.
Each model input name needs to correspond to a field in the pydantic model, and the type of the
field needs to make sense given the shape and datatype of the input. In this case, there
is just a single input named "text" that is supposed to hold a single text, so we simply define
```python
from pydantic_open_inference import InputsBaseModel

class ExampleModelInputs(InputsBaseModel):
    text: str

```

In the same way, we define a pydantic model for the outputs, inheriting from `OutputsBaseModel`.
There is one model outputs named "entities", so that will be the only field we define.
There can be any number of entities (corresponding to the -1 in the shape), so we define it to
be a `list`. A list of what? Here we have some choices. We could simply define `entities: list[list[str]]`
and get each entity as a list of strings, e.g., `["order-id", "0.98", "12", "24"]`, but this is not
very useful to us, so we instead define, e.g., a _named tuple_ with ordered, named, and typed fields:
```python
from typing import NamedTuple
from enum import StrEnum
from pydantic_open_inference import OutputsBaseModel

class Label(StrEnum):
    ORDER_ID = "order-id"
    INVOICE_ID = "invoice-id"

class Entity(NamedTuple):
    label: Label
    score: float
    start: int
    end: int

class ExampleModelOutputs(OutputsBaseModel):
    entities: list[Entity]
```
Note that we could also have used `typing.Literal` instead of `enum.StrEnum`, and even added further validation,
e.g., that the score is between 0.0 and 1.0, and that the start index is not negative etc. Pydantic offers many
possibilities, but for now, let us stick with this.

We now create a `RemoteModel` instance. The `RemoteModel` class represents a model in the inference server,
in this case our "example_model". Instantiation requires the URL of the inference server, the name of the
model, and the pydantic models we defined for the input and output:
```python
from pydantic_open_inference import RemoteModel

INFERENCE_SERVER_URL = "http://localhost:8080"

example_model = RemoteModel(
    server_url=INFERENCE_SERVER_URL,
    model_name="example_model",
    inputs_model=ExampleModelInputs,
    outputs_model=ExampleModelOutputs,
)
```

In order to call the inference server, we simply call the `RemoteModel.infer` method, giving
it an instance of `ExampleModelInputs`, and get back an instance of `ExampleModelOutputs`:
```python
example_input = ExampleModelInputs(text="Your order id is 123456.")
example_output = example_model.infer(example_input)
# ExampleModelOutputs(
#    entities=[
#        Entity(label=Label.ORDER_ID, score=0.95, start=17, end=23),
#    ]
# )
```
We have thus not only made a perfectly type-safe and validated call to the inference server,
we have also gotten the response back in a useful format! We never had to bother with
handling the shape of the data, parsing it in row-major order etc, as all of that is handled under the hood!

The only time you need to think about "shape" is when defining the fields of the pydantic models.
Use types like, e.g., `list`, `tuple`, and `typing.NamedTuple`, to type the pydantic-model fields
so that they map sensibly to their corresponding model input/output. Each input/output name must
match to a pydantic field and the shape must correspond to the structure of the field, e.g.,
`list[tuple[int, int]]` for shape [-1, 2], or `tuple[tuple[float, float], tuple[float, float]]`
for shape [2, 2], etc.

The "datatype" of the model _inputs_ is automatically set based on the innermost type of the
corresponding field in the `InputsBaseModel` subclass, according to:

| python type | datatype |
|-------------|----------|
| bool        | BOOL     |
| int         | INT64    |
| float       | FP32     |
| str         | BYTES    |

However, you can always _override_ the datatype of the model inputs to whatever you want using
`pydantic_open_inference.DatatypeOverride`; see its docstring for more information.
The datatypes of the model _outputs_ are disregarded; all that matters is 
that the data we got back from the model can be mapped to the defined `OutputsBaseModel` subclass.

When an inference call to a model is made, it automatically requests only the specific outputs
for which you have defined fields in your `OutputsBaseModel` subclass. So if you are not
interested in an output, simply do not define a pydantic field for it.

## Examples

See the [examples directory](examples) for more examples.
