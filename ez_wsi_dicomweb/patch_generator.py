# Copyright 2023 Google LLC
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
"""An abstraction to sample patches in a slide at a specified pixel spacing."""

import dataclasses
from typing import Iterator, Optional

import cv2
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import pixel_spacing
import numpy as np

TISSUE_MASK_PIXEL_SPACING = pixel_spacing.PixelSpacing.FromMagnificationString(
    "1.250X"
)


@dataclasses.dataclass(frozen=True)
class StrideCoordinate:
  """Represents the coordinates of a stride using x, y coordinates.

  See PatchGenerator for usage examples.

  Attributes:
    y_strides: The number of strides along the y axis
    x_strides: The number of strides along the x axis
  """

  y_strides: int
  x_strides: int


class PatchGenerator:
  """A generator for patch sampling in a slide at specified pixel spacing.

  The sampling process moves a sliding window of size patch_size with
  steps of size stride_size. The first sampled patch is centered at
  (-stride_size / 2, -stride_size / 2). We use (y_strides, x_strides) to denote
  how many strides we have taken along the y and x axis. Given a specified
  (y_strides, x_strides), a sampled patch can be uniquely identified and will
  be centered at (-stride_size / 2 + x_strides * stride_size,
  -stride_size / 2 + y_strides * stride_size).

  Utility functions are also provided to translate between strides and
  patch coordinates.
  """

  def __init__(
      self,
      slide: dicom_slide.DicomSlide,
      ps: pixel_spacing.PixelSpacing,
      stride_size: int,
      patch_size: int,
      max_luminance: float = 0.8,
      normalized_tissue_mask: Optional[np.ndarray] = None,
      tissue_mask_pixel_spacing: pixel_spacing.PixelSpacing = TISSUE_MASK_PIXEL_SPACING,
  ):
    """Constructor.

    Args:
      slide: The slide to generate on.
      ps: The pixel spacing that we sample patches on.
      stride_size: Stride size at the pixel spacing that we sample patches.
      patch_size: Patch size at the pixel spacing that we sample patches.
      max_luminance: Regions with luminance (grayscale) > this threshold are to
        be considered non-tissue background, and will be discarded in the patch
        sampling.
      normalized_tissue_mask: If provided, will be cached as the tissue mask.
      tissue_mask_pixel_spacing: Used to determine where tissue is present if
        provided. If the provided DICOMSlide does not contain the provided pixel
        spacing, patch generation will fail. This parameter must be overidden if
        the default pixel spacing is not avaliable.
    """
    self.slide = slide
    self.pixel_spacing = ps
    self.stride_size = stride_size
    self.patch_size = patch_size
    self.max_luminance = max_luminance
    self._normalized_tissue_mask_cache = normalized_tissue_mask
    self._tissue_mask_pixel_spacing = tissue_mask_pixel_spacing

  def __iter__(self) -> Iterator[dicom_slide.Patch]:
    """Given tissue mask, emits patches.

    Yields:
      A sampled patch.

    Raises:
      DicomImageMissingRegionError if there is no region with the
      required luminance values.
    """
    y_coords, x_coords = np.where(self._normalized_tissue_mask())
    for y, x in zip(y_coords, x_coords):
      yield self.patch_at_stride(StrideCoordinate(y, x))

  def _normalized_tissue_mask(self):
    """An image mask that is likely to contain tissues.

    It is normalized/down-scaled by the specified stride size, i.e.
    normalized_tissue_mask size = original size / stride.

    Returns:
      A 2D ndarray binary (0, 1) image denoting tissue mask.

    Raises:
      DicomImageMissingRegionError if there is no region with the
      required luminance values.
    """
    if self._normalized_tissue_mask_cache is not None:
      return self._normalized_tissue_mask_cache

    rgb_at_tissue_ps = self.slide.get_image(
        self._tissue_mask_pixel_spacing
    ).image_bytes()
    # luminance values
    if np.issubdtype(rgb_at_tissue_ps.dtype, np.floating):
      if rgb_at_tissue_ps.max() <= 1.0:
        rgb_at_tissue_ps *= 255.0
      rgb_at_tissue_ps = rgb_at_tissue_ps.astype(np.uint8)
    gray_tissue_mask = cv2.cvtColor(rgb_at_tissue_ps, cv2.COLOR_RGB2GRAY)
    tissue_ps_stride_size = (
        self._tissue_mask_pixel_spacing.pixel_spacing_mm
        / self.pixel_spacing.pixel_spacing_mm
    )

    scaling_factor = self.stride_size / tissue_ps_stride_size
    new_dimensions = (
        max(int(gray_tissue_mask.shape[1] // scaling_factor), 1),
        max(int(gray_tissue_mask.shape[0] // scaling_factor), 1),
    )
    gray_image_at_output_res = cv2.resize(
        gray_tissue_mask, new_dimensions, interpolation=cv2.INTER_AREA
    )
    self._normalized_tissue_mask_cache = (gray_image_at_output_res > 0) & (
        gray_image_at_output_res <= int(255.0 * self.max_luminance)
    )
    if np.all(~self._normalized_tissue_mask_cache):  # pylint: disable=invalid-unary-operand-type
      raise ez_wsi_errors.DicomImageMissingRegionError(
          f"Slide with study_uid {self.slide.path.study_uid} "
          f"series_uid {self.slide.path.series_uid} has no region with "
          "luminance value less than threshold "
          f"{self.max_luminance}."
      )
    return self._normalized_tissue_mask_cache

  def total_num_patches(self) -> int:
    """Total number of tissue patches in this slide."""
    y, _ = np.where(self._normalized_tissue_mask())
    return len(y)

  def total_strides(self) -> StrideCoordinate:
    y, x = self._normalized_tissue_mask().shape
    return StrideCoordinate(y_strides=y, x_strides=x)

  def strides_to_patch_bounds(
      self, strides: StrideCoordinate
  ) -> dicom_slide.PatchBounds:
    """Converts number of strides to patch coordinates.

    Args:
      strides: Number of strides taken on x and y axis.

    Returns:
      Patch Bounds corresponding to the strides.
    """
    center_y = self.stride_size * strides.y_strides + self.stride_size // 2
    center_x = self.stride_size * strides.x_strides + self.stride_size // 2
    top = center_y - self.patch_size // 2
    left = center_x - self.patch_size // 2

    return dicom_slide.PatchBounds(
        x_origin=left,
        y_origin=top,
        width=self.patch_size,
        height=self.patch_size,
    )

  def patch_bounds_to_strides(
      self, patch_bounds: dicom_slide.PatchBounds
  ) -> StrideCoordinate:
    """Converts patch bounds to number of strides in tissue mask.

    Args:
      patch_bounds: Describes the patch's coordinates bounding box.

    Returns:
      Corresponding number of strides.
    """
    half_stride = self.stride_size // 2
    half_patch_size = self.patch_size // 2
    center_y = patch_bounds.y_origin + half_patch_size - half_stride
    center_x = patch_bounds.x_origin + half_patch_size - half_stride
    y = center_y // self.stride_size
    x = center_x // self.stride_size
    return StrideCoordinate(y_strides=y, x_strides=x)

  def patch_at_stride(self, strides: StrideCoordinate) -> dicom_slide.Patch:
    """Converts tissue mask coordinates to patch coordinates.

    Args:
      strides: Number of strides taken from x and y axis.

    Returns:
      Corresponding Patch.
    """
    patch_bounds = self.strides_to_patch_bounds(strides)

    return self.slide.get_patch(
        self.pixel_spacing,
        x=patch_bounds.x_origin,
        y=patch_bounds.y_origin,
        width=patch_bounds.width,
        height=patch_bounds.height,
    )
