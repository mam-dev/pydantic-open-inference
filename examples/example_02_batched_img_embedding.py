"""Example: Call an image embedding model in an open-inference server.

The model inputs consist of a batch of images, each image consisting of 150528 floats
between 0.0 and 1.0. The model outputs consist of a corresponding batch of
embeddings, each consisting of 512 floats.

"""

from typing import Annotated

import pydantic

from pydantic_open_inference import (
    InputsBaseModel,
    OutputsBaseModel,
    RemoteModel,
)

NR_OF_IMAGE_VALUES = 150528
NR_OF_EMBEDDINGS = 512

# Image, 150528 (224x224x3, flattened) floats between 0.0 and 1.0
ImageValue = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]
Image = Annotated[list[ImageValue], pydantic.Field(min_length=NR_OF_IMAGE_VALUES, max_length=NR_OF_IMAGE_VALUES)]
Embedding = Annotated[list[float], pydantic.Field(min_length=NR_OF_EMBEDDINGS, max_length=NR_OF_EMBEDDINGS)]


class ImageEmbeddingInput(InputsBaseModel):
    """Input for an image embedding model that processes batches of images.

    Assumes preprocessed images flattened to 1D arrays.
    """

    # Batch of images, each 224x224x3, flattened
    # Shape: [-1, 150528] (224*224*3 = 150528)
    images: list[Image]


class ImageEmbeddingOutput(OutputsBaseModel):
    """Output embeddings from the model."""

    # Batch of embeddings, each 512-dimensional
    # Shape: [-1, 512]
    embeddings: list[Embedding]


image_embedder = RemoteModel(
    model_name="resnet50_embeddings",
    inputs_model=ImageEmbeddingInput,
    outputs_model=ImageEmbeddingOutput,
    server_url="http://triton-server:8000",
    model_version="1",  # Optional: specify model version
)

# Example with dummy data
batch_images = [
    [0.5] * 150528,  # Image 1 (dummy flattened pixel values)
    [0.3] * 150528,  # Image 2
    [0.7] * 150528,  # Image 3
    [0.2] * 150528,  # Image 4
]

input_batch = ImageEmbeddingInput(images=batch_images)
embeddings_result = image_embedder.infer(input_batch)

print(f"Generated {len(embeddings_result.embeddings)} embeddings")
print(f"Each embedding has dimension: {len(embeddings_result.embeddings[0])}")
