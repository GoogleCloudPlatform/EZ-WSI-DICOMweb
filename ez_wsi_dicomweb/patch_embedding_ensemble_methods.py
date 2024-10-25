# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods to generate embeddings for patches dim >= embedding input dim."""
import abc
import enum
from typing import Iterator, Sequence, Tuple, Union
import uuid

from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import patch_embedding_types
import numpy as np


class SinglePatchEnsemblePosition(enum.Enum):
  UPPER_LEFT = 'UPPER_LEFT'
  UPPER_RIGHT = 'UPPER_RIGHT'
  CENTER = 'CENTER'
  LOWER_LEFT = 'LOWER_LEFT'
  LOWER_RIGHT = 'LOWER_RIGHT'


_ReducedType = Union[
    Sequence[patch_embedding_types.EmbeddingResult],
    Sequence[patch_embedding_types.PatchEmbeddingEnsembleResult],
]


class PatchEnsembleMethod(metaclass=abc.ABCMeta):
  """Defines operation to define & combine regions of patch to gen embeddings."""

  def __init__(self):
    self._ensemble_id_base = str(uuid.uuid4())
    self._ensemble_id = 1

  def _get_ensemble_id(self) -> str:
    val = self._ensemble_id
    self._ensemble_id += 1
    return f'{self._ensemble_id_base}-{val}'

  def _validate_patch_dimensions(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> None:
    """Validates that patch position and input dimensions are valid.

    Args:
      endpoint: Patch embedding endpoint.
      patch: Patch to embedd.

    Raises:
      ez_wsi_errors.PatchEmbeddingDimensionError: Patch input falls outside
        of slide image or patch dimensions are <= endpoint input dimensions.
    """
    # Test patch dimensions >= embedding endpoint input dimensions.
    if (
        endpoint.patch_width() > patch.width
        or endpoint.patch_height() > patch.height
    ):
      raise ez_wsi_errors.PatchEmbeddingDimensionError(
          f'Patch dimensions ({patch.width}, {patch.height}) are less than '
          f' embedding input dimensions ({endpoint.patch_width()},'
          f' {endpoint.patch_height()}).'
      )
    if patch.x < 0 or patch.y < 0 or patch.width <= 0 or patch.height <= 0:
      raise ez_wsi_errors.PatchEmbeddingDimensionError(
          f'Invalid patch dimensions ({patch.x}, {patch.y} to'
          f' {patch.x + patch.width -1}, {patch.y  + patch.height -1} ).'
      )

  @abc.abstractmethod
  def generate_ensemble(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
    """Yields iterator of patches of embedding dim to gen embedding for patch.

    Args:
      endpoint: Embedding endpoint used to generate patch embeddings.
      patch: Input pixel region to generate an embedding.

    Yields:
      PatchEmbeddingSource that define one or more sub patches that are
      required to generate an embedding for the patch.
    """

  @abc.abstractmethod
  def reduce_ensemble(
      self,
      patch: patch_embedding_types.EmbeddingPatch,
      ensemble_list: _ReducedType,
  ) -> patch_embedding_types.EmbeddingResult:
    """Returns single embedding result from ensemble of patch embeddings.

    Args:
      patch: Input pixel region embedding was generated from
      ensemble_list: List of embedding results generated within patch

    Returns:
      Single embedding result for patch.
    """


def _raise_if_error(
    ensemble_result: Union[
        patch_embedding_types.EmbeddingResult,
        patch_embedding_types.PatchEmbeddingEnsembleResult,
    ],
) -> None:
  """Raises if patch_embedding_types.PatchEmbeddingEnsembleResult has error."""
  if (
      isinstance(
          ensemble_result, patch_embedding_types.PatchEmbeddingEnsembleResult
      )
      and ensemble_result.error is not None
  ):
    raise ez_wsi_errors.PatchEmbeddingEndpointError(
        ensemble_result.error.error_message
    )


def _get_sub_patch_position(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    patch: patch_embedding_types.EmbeddingPatch,
    position: SinglePatchEnsemblePosition,
) -> Tuple[int, int]:
  """Return sub_patch position within a patch defined by enum."""
  if position == SinglePatchEnsemblePosition.UPPER_LEFT:
    pos_x, pos_y = patch.x, patch.y
  elif position == SinglePatchEnsemblePosition.UPPER_RIGHT:
    pos_x, pos_y = patch.x + patch.width - endpoint.patch_width(), patch.y
  elif position == SinglePatchEnsemblePosition.LOWER_LEFT:
    pos_x, pos_y = patch.x, patch.y + patch.height - endpoint.patch_height()
  elif position == SinglePatchEnsemblePosition.LOWER_RIGHT:
    pos_x = patch.x + patch.width - endpoint.patch_width()
    pos_y = patch.y + patch.height - endpoint.patch_height()
  elif position == SinglePatchEnsemblePosition.CENTER:
    pos_x = int(patch.x + (patch.width - endpoint.patch_width()) / 2)
    pos_y = int(patch.y + (patch.height - endpoint.patch_height()) / 2)
  else:
    raise ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError(
        'Invalid SinglePatchEnsemblePosition.'
    )
  pos_x = int(
      max(min(pos_x, patch.x + patch.width - endpoint.patch_width()), patch.x)
  )
  pos_y = int(
      max(
          min(pos_y, patch.y + patch.height - endpoint.patch_height()),
          patch.y,
      )
  )
  return pos_x, pos_y


class SinglePatchEnsemble(PatchEnsembleMethod):
  """Returns embedding generated from a single patch."""

  def __init__(self, position: SinglePatchEnsemblePosition):
    """SinglePatchEnsemble Constructor.

    Args:
      position: Position of patch to generate embedding.

    Raises:
      ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError: Invalid
        SinglePatchEnsemblePosition.
    """
    super().__init__()
    self._position = position
    try:
      if position not in SinglePatchEnsemblePosition:
        raise ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError(
            'Invalid SinglePatchEnsemblePosition.'
        )
    except TypeError as e:
      raise ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError(
          'Invalid SinglePatchEnsemblePosition.'
      ) from e

  def generate_ensemble(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
    """Yields iterator of patches of embedding dim to gen embedding for patch.

    Args:
      endpoint: Embedding endpoint used to generate patch embeddings.
      patch: Input pixel region to generate an embedding.

    Yields:
      PatchEmbeddingSource that define one or more sub patches that are
      required to generate an embedding for the patch.

    Raises:
      ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError: Invalid
        SinglePatchEnsemblePosition.
      ez_wsi_errors.PatchEmbeddingDimensionError: Patch input falls outside
        of slide image or patch dimensions are <= endpoint input dimensions.
    """
    self._validate_patch_dimensions(endpoint, patch)

    ensemble_id = self._get_ensemble_id()
    pos_x, pos_y = _get_sub_patch_position(endpoint, patch, self._position)
    yield patch_embedding_types.PatchEmbeddingSource(
        patch.get_patch(
            pos_x,
            pos_y,
            endpoint.patch_width(),
            endpoint.patch_height(),
        ),
        patch,
        ensemble_id,
    )

  def reduce_ensemble(
      self,
      patch: patch_embedding_types.EmbeddingPatch,
      ensemble_list: _ReducedType,
  ) -> patch_embedding_types.EmbeddingResult:
    """Returns single embedding result from ensemble of patch embeddings.

    Args:
      patch: Input pixel region embedding was generated from
      ensemble_list: List of embedding results generated within patch

    Returns:
      Single embedding result for patch.

    Raises:
      ez_wsi_errors.SinglePatchEmbeddingEnsembleError: Ensemble results did not
        retuurn one embedding.
    """
    if len(ensemble_list) != 1:
      raise ez_wsi_errors.SinglePatchEmbeddingEnsembleError(
          'SinglePatchEnsemble requires exactly one embedding result.'
      )
    _raise_if_error(ensemble_list[0])
    return patch_embedding_types.EmbeddingResult(
        patch, ensemble_list[0].embedding
    )


class DefaultSinglePatchEnsemble(SinglePatchEnsemble):
  """Returns single embedding for patch, validates patch dim = embedding dim."""

  def __init__(self):
    super().__init__(SinglePatchEnsemblePosition.UPPER_LEFT)

  def generate_ensemble(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
    if (
        endpoint.patch_width() != patch.width
        or endpoint.patch_height() != patch.height
    ):
      raise ez_wsi_errors.PatchEmbeddingDimensionError(
          f'Patch dimensions ({patch.width}, {patch.height}) do not match'
          f' endpoint embedding input dimensions ({endpoint.patch_width()},'
          f' {endpoint.patch_height()}). A patch ensable method must be defined'
          ' to generate patches, e.g., MeanPatchEmbeddingEnsemble or'
          ' SinglePatchEnsemble.'
      )
    return super().generate_ensemble(endpoint, patch)


class FivePatchMeanEnsemble(PatchEnsembleMethod):
  """Returns mean embedding from five patches sampled across the patch."""

  def generate_ensemble(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
    self._validate_patch_dimensions(endpoint, patch)
    ensemble_id = self._get_ensemble_id()
    endpoint_width = endpoint.patch_width()
    endpoint_height = endpoint.patch_height()
    if patch.width == endpoint_width and patch.height == endpoint_height:
      # if patches overlap perfectly, just yield one.
      sampling_positions = [SinglePatchEnsemblePosition.UPPER_LEFT]
    else:
      sampling_positions = [
          SinglePatchEnsemblePosition.UPPER_LEFT,
          SinglePatchEnsemblePosition.UPPER_RIGHT,
          SinglePatchEnsemblePosition.CENTER,
          SinglePatchEnsemblePosition.LOWER_LEFT,
          SinglePatchEnsemblePosition.LOWER_RIGHT,
      ]
    for position in sampling_positions:
      pos_x, pos_y = _get_sub_patch_position(endpoint, patch, position)
      yield patch_embedding_types.PatchEmbeddingSource(
          patch.get_patch(
              pos_x,
              pos_y,
              endpoint_width,
              endpoint_height,
          ),
          patch,
          ensemble_id,
      )

  def reduce_ensemble(
      self,
      patch: patch_embedding_types.EmbeddingPatch,
      ensemble_list: _ReducedType,
  ) -> patch_embedding_types.EmbeddingResult:
    if not ensemble_list:
      raise ez_wsi_errors.MeanPatchEmbeddingEnsembleError(
          'MeanPatchEmbeddingEnsemble requires at least one embedding result.'
      )
    first_result = ensemble_list[0]
    _raise_if_error(first_result)
    embedding_dtype = first_result.embedding.dtype
    dtype_cast = (
        not np.issubdtype(embedding_dtype, np.floating)
        or embedding_dtype.itemsize < 8
    )
    embedding = np.zeros(
        first_result.embedding.shape,
        dtype=np.float64 if dtype_cast else embedding_dtype,
    )
    for result in ensemble_list:
      embedding += result.embedding
    embedding /= float(len(ensemble_list))
    if dtype_cast:
      embedding = embedding.astype(embedding_dtype)
    return patch_embedding_types.EmbeddingResult(patch, embedding)


class MeanPatchEmbeddingEnsemble(FivePatchMeanEnsemble):
  """Returns mean embedding from set of embeddings sampled across the patch."""

  def __init__(self, step_x_px: int, step_y_px: int):
    """MeanPatchEmbeddingEnsemble Constructor.

    Args:
      step_x_px: Step size in x direction to sample patch for embedding.
      step_y_px: Step size in y direction to sample patch for embedding.

    Raises:
      ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError: Invalid
        step size.
    """
    super().__init__()
    self._step_x = step_x_px
    self._step_y = step_y_px
    if self._step_x <= 0 or self._step_y <= 0:
      raise ez_wsi_errors.MeanPatchEmbeddingEnsembleError(
          'MeanPatchEmbeddingEnsemble requires a minimum of 1 px patch step.'
      )

  def generate_ensemble(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patch: patch_embedding_types.EmbeddingPatch,
  ) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
    self._validate_patch_dimensions(endpoint, patch)
    embedding_width = endpoint.patch_width()
    embedding_height = endpoint.patch_height()
    start_x = patch.x
    start_y = patch.y
    end_x = max(start_x, start_x + patch.width - embedding_width)
    end_y = max(start_y, start_y + patch.height - embedding_height)
    ensemble_id = self._get_ensemble_id()
    for y in range(start_y, end_y, self._step_y):
      for x in range(start_x, end_x, self._step_x):
        yield patch_embedding_types.PatchEmbeddingSource(
            patch.get_patch(
                x,
                y,
                embedding_width,
                embedding_height,
            ),
            patch,
            ensemble_id,
        )


def mean_patch_embedding(
    embeddings: Union[
        Iterator[patch_embedding_types.EmbeddingResult],
        Sequence[patch_embedding_types.EmbeddingResult],
    ],
) -> np.ndarray:
  """Returns mean embedding from list of or iterator of embedding results."""
  if isinstance(embeddings, Sequence):
    embeddings = iter(embeddings)
  try:
    result = next(embeddings)
  except StopIteration:
    raise ez_wsi_errors.MeanPatchEmbeddingEnsembleError(
        'MeanPatchEmbeddingEnsemble requires at least one embedding result.'
    ) from None
  embedding_dtype = result.embedding.dtype
  dtype_cast = (
      not np.issubdtype(embedding_dtype, np.floating)
      or embedding_dtype.itemsize < 8
  )
  embedding = np.zeros(
      result.embedding.shape,
      dtype=np.float64 if dtype_cast else embedding_dtype,
  )
  embedding += result.embedding
  count = 1
  for result in embeddings:
    embedding += result.embedding
    count += 1
  embedding /= float(count)
  if dtype_cast:
    embedding = embedding.astype(embedding_dtype)
  return embedding
