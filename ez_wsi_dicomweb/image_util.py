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
"""Utility functions for image processing."""

import io
from typing import Optional, Union

from ez_wsi_dicomweb import ez_wsi_errors
import numpy as np
import PIL
from PIL import ImageCms

_RGB = "RGB"


def get_image_bytes_samples_per_pixel(image_bytes: np.ndarray) -> int:
  """Returns the number of samples per pixel in the image.

  Args:
    image_bytes: Uncompressed image bytes (e.g., 8 bit RGB)

  Raises:
    ez_wsi_errors.GcsImageError: If the image is not 2D or 3D.
  """
  if len(image_bytes.shape) == 2:
    return 1
  elif len(image_bytes.shape) == 3:
    return image_bytes.shape[2]
  raise ez_wsi_errors.GcsImageError(
      f"Invalid image shape: {image_bytes.shape}. Image must be 2D or 3D."
  )


def create_icc_profile_transformation(
    source_image_icc_profile_bytes: bytes,
    dest_icc_profile: Union[bytes, ImageCms.core.CmsProfile],
    rendering_intent: ImageCms.Intent = ImageCms.Intent.PERCEPTUAL,
) -> Optional[ImageCms.ImageCmsTransform]:
  """Returns transformation to from pyramid colorspace to icc_profile.

  Args:
    source_image_icc_profile_bytes: Source image icc profile bytes.
    dest_icc_profile: ICC Profile to DICOM Pyramid imaging to.
    rendering_intent: Rendering intent to use in transformation.

  Returns:
    PIL.ImageCmsTransformation to transform pixel imaging or None.
  """
  if not source_image_icc_profile_bytes or not dest_icc_profile:
    return None
  dicom_input_profile = ImageCms.getOpenProfile(
      io.BytesIO(source_image_icc_profile_bytes)
  )
  if isinstance(dest_icc_profile, bytes):
    dest_icc_profile = ImageCms.getOpenProfile(io.BytesIO(dest_icc_profile))
  return ImageCms.buildTransform(
      dicom_input_profile,
      dest_icc_profile,
      _RGB,
      _RGB,
      renderingIntent=rendering_intent,
  )


def transform_image_bytes_color(
    image_bytes: np.ndarray,
    color_transform: Optional[ImageCms.ImageCmsTransform] = None,
) -> np.ndarray:
  """Transforms image bytes color using ICC Profile Transformation."""
  samples_per_pixel = get_image_bytes_samples_per_pixel(image_bytes)
  if color_transform is None or samples_per_pixel <= 1:
    return image_bytes
  with PIL.Image.fromarray(image_bytes) as img:
    ImageCms.applyTransform(img, color_transform, inPlace=True)
    return np.asarray(img)
