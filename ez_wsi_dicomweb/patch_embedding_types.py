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
"""Types for patch embedding."""

import dataclasses
import math
import typing
from typing import List, Optional, Union

from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import slide_level_map
import numpy as np

EmbeddingPatch = Union[dicom_slide.DicomPatch, gcs_image.GcsPatch]


@dataclasses.dataclass(frozen=True)
class PatchEmbeddingSource:
  """Defines patch embedding source."""

  patch: EmbeddingPatch
  ensemble_source_patch: EmbeddingPatch
  ensemble_id: str

  @property
  def mag_scaled_embedding_patch_count(self) -> int:
    """Returns number of patch request count. Scales count by downsamping mag.

    Pete endpoint limits total patch requests and scales DICOM patch requests by
    magnification to estimate backend load. Perform similar scaling here to
    keep client and server in sync.

    Returns:
      request count size for patch.
    """
    patch = self.patch
    if not isinstance(patch, dicom_slide.DicomPatch):
      return 1
    if not isinstance(patch.level, slide_level_map.ResizedLevel):
      return 1
    resize_level = patch.level
    source_level = resize_level.source_level
    patch_count_scale_factor = math.ceil(max(resize_level.scale_factors())) ** 2
    return min(patch_count_scale_factor, source_level.number_of_frames)


@dataclasses.dataclass(frozen=True)
class SlideEmbeddingSource:
  """Defines List of Patch embedding from the same source."""

  patches: List[PatchEmbeddingSource]

  @property
  def mag_scaled_embedding_patch_count(self) -> int:
    """Returns number 'patches' for wsi patch count is scaled by mag factor."""
    return sum(p.mag_scaled_embedding_patch_count for p in self.patches)

  def get_bearer_token(self) -> str:
    """Returns bearer token used to access patches."""
    headers = self.patches[0].patch.get_gcp_data_credential_header()
    try:
      return headers['authorization'].split('Bearer ')[-1].strip()
    except (KeyError, IndexError) as _:
      return ''


@dataclasses.dataclass(frozen=True)
class PatchEmbeddingError:
  error_code: str
  error_message: str


@dataclasses.dataclass(frozen=True)
class PatchEmbeddingEnsembleResult:
  """Embedding for part of an ensemble used to define a patch embedding."""

  input_patch: PatchEmbeddingSource
  _embedding: Optional[np.ndarray]
  error: Optional[PatchEmbeddingError]

  def __post_init__(self):
    if (self._embedding is None and self.error is None) or (
        self._embedding is not None and self.error is not None
    ):
      raise ValueError('Internal error')

  @property
  def embedding(self) -> np.ndarray:
    if self._embedding is not None:
      return self._embedding
    error = typing.cast(PatchEmbeddingError, self.error)
    raise ez_wsi_errors.PatchEmbeddingEndpointError(error.error_message)


@dataclasses.dataclass(frozen=True)
class EmbeddingResult:
  """Embedding result for a patch."""

  patch: EmbeddingPatch
  embedding: np.ndarray
