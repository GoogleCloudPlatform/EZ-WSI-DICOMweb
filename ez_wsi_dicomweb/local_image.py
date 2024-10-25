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
"""Creates Images and Patches from tradtional image formats stored locally."""
from __future__ import annotations

import io
from typing import BinaryIO, Optional, Union

from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import gcs_image
import numpy as np


ImageDimensions = gcs_image.ImageDimensions
LocalImageSourceTypes = Union[np.ndarray, BinaryIO, bytes, str]


class LocalImage(gcs_image.GcsImage):
  """Create patches from images stored on localsystem."""

  def __init__(
      self,
      image_source: LocalImageSourceTypes,
      image_dimensions: Optional[ImageDimensions] = None,
  ):
    """Initializes the LocalImage.

    LocalImage represents a image in tradtional image format (PNG, JPEG, etc)
    that is stored on the local file system or in memory

    Args:
      image_source: Image source can be a str representing a style path to an
        image on a local file system; or an readable file like object that
        contains compressed image bytes for a traditional image format, e.g. PNG
        or jpeg bytes; bytes that contain a compressed image (PNG, JPEG, etc);
        or a numpy array containing uncompressed RGB or single channel image.
      image_dimensions: Image dimensions to resize the source image to.

    Raises:
      GcsImageError: If image source is not supported.
    """
    if isinstance(image_source, str):
      self._filename = image_source
      with open(image_source, 'rb') as f:
        image_source = f.read()
    elif isinstance(image_source, io.IOBase):
      self._filename = ''
      image_source = image_source.read()
    elif isinstance(image_source, bytes):
      self._filename = ''
    else:
      self._filename = ''
    super().__init__(
        image_source,
        image_dimensions=image_dimensions,
        credential_factory=credential_factory.NoAuthCredentialsFactory(),
    )
    if self._filename:
      # If image is loaded from file on the file system don't store compressed
      # bytes assume it will be more optimal for to re-read bytes from FS if
      # it is optimal for the embeedding API to retrieve embeddings from the
      # whole image.
      self.clear_source_image_compressed_bytes()

  @property
  def filename(self) -> str:
    return self._filename

  def _get_source_image_bytes_from_file(self) -> bytes:
    if not self._filename:
      return b''
    try:
      with open(self._filename, 'rb') as f:
        return f.read()
    except OSError:
      return b''
