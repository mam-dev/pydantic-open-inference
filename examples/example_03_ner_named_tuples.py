"""Example: Use named tuples to handle structured data."""

from typing import NamedTuple

from pydantic_open_inference import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)


class NERInput(InputsBaseModel):
    """Input for NER model."""

    text: str
    max_entities: int  # Maximum number of entities to return


class Entity(NamedTuple):
    """A named entity from the model."""

    text: str
    label: str
    score: float


class NEROutput(OutputsBaseModel):
    """Output from NER model - list of entities with structured data."""

    # Each entity has: text (str), label (str), score (float)
    # Shape: [N, 3] where N is number of entities
    entities: list[Entity]


ner_model = RemoteModel(
    model_name="ner_model",
    inputs_model=NERInput,
    outputs_model=NEROutput,
    server_url="http://localhost:8000",
)

ner_input = NERInput(
    text="Apple Inc. was founded by Steve Jobs in California.",
    max_entities=10,
)
ner_result = ner_model.infer(ner_input)
for entity in ner_result.entities:
    print(f"Entity: '{entity.text}' | Label: {entity.label} | Score: {entity.score:.3f}")
