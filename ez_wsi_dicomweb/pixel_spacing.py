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
"""A class that represents a WSI slide's pixel spacing."""
from __future__ import annotations

import math

from ez_wsi_dicomweb import ez_wsi_errors


# The tolerance (percentage difference) for difference between pixel spacings.
# EZ WSI expects practically square pixels.
PIXEL_SPACING_DIFF_TOLERANCE = 0.01
# Magnification & Pixel Spacing are usually linearly related via a constant
# scaling factor of 0.01. he imaging scale factor can be directly derived via a
# single picture of a microscope calibration slide captured at a known
# magnification.
_SCALE_FACTOR = 0.01


class PixelSpacing:
  """Represents the pixel spacing of a DICOM/WSI slide.

  NOTE: By default the two provided spacings are averaged & used to compute a
    pixel spacing attribute that is used internally. This is because EZ WSI
    assumes square pixels.

  See also: https://dicom.innolitics.com/ciods/rt-dose/image-plane/00280030.
  """

  def __init__(
      self,
      row_spacing: float,
      column_spacing: float,
      scaling_factor: float = _SCALE_FACTOR,
      spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE,
  ):
    """Constructor.

    Args:
      row_spacing: a number representing the pixel spacing measured vertically
        in mm/px.
      column_spacing: a number representing the pixel spacing measured
        horizontally in mm/px.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.
      spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.

    Raises:
      NonSquarePixelError if the provided pixels are not roughly square.
    """

    if not math.isclose(
        row_spacing, column_spacing, rel_tol=spacing_diff_tolerance
    ):
      raise ez_wsi_errors.NonSquarePixelError(
          f'The provided row_spacing {row_spacing} and column_spacing'
          f' {column_spacing} are not square pixels.'
      )

    self._row_spacing = row_spacing
    self._column_spacing = column_spacing
    self._pixel_spacing = (row_spacing + column_spacing) / 2.0
    self._scaling_factor = scaling_factor
    self._spacing_diff_tolerance = spacing_diff_tolerance

  @classmethod
  def FromString(
      cls, pixel_spacing: str, scaling_factor: float = _SCALE_FACTOR
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a string representing a pixel spacing measured in in mm/px creates a
    PixelSpacing object.

    Args:
      pixel_spacing: The string value of the pixel spacing, e.g. a string
        representing a double measured in in mm/px.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.

    Returns:
      A new PixelSpacing object.
    """
    return PixelSpacing.FromDouble(
        pixel_spacing=float(pixel_spacing), scaling_factor=scaling_factor
    )

  @classmethod
  def FromMagnificationString(
      cls, magnification: str, scaling_factor: float = _SCALE_FACTOR
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a string representing a Magnificiation level creates a PixelSpacing
    object.

    Args:
      magnification: The string value of the magnification, e.g. a string
        representing a zoom level: 5, 5X, 10, 10X, 25X.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.

    Returns:
      A new PixelSpacing object.

    Raises:
      InvalidMagnificationStringError if the magnfication is not a float or <=0.
    """
    magnification = magnification.lower().rstrip('x')

    if not magnification.replace('.', '').isnumeric():
      raise ez_wsi_errors.InvalidMagnificationStringError(
          f'Provided magnification {magnification} is not a float.'
      )

    if float(magnification) <= 0:
      raise ez_wsi_errors.InvalidMagnificationStringError(
          f'Provided magnification {magnification} is <= 0. Magnifications must'
          ' be positive.'
      )

    pixel_spacing = scaling_factor / float(magnification)

    return PixelSpacing.FromDouble(
        pixel_spacing=pixel_spacing, scaling_factor=scaling_factor
    )

  @classmethod
  def FromDouble(
      cls, pixel_spacing: float, scaling_factor: float = _SCALE_FACTOR
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a pixel spacing float measured in mm/px creates a PixelSpacing object.

    Args:
      pixel_spacing: The avg value of the row & column pixel spacing in mm/px.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.

    Returns:
      A new PixelSpacing object.
    """
    return PixelSpacing(
        row_spacing=pixel_spacing,
        column_spacing=pixel_spacing,
        scaling_factor=scaling_factor,
    )

  def __hash__(self):
    return hash(
        tuple(
            (self.row_spacing_mm, self.column_spacing_mm, self.pixel_spacing_mm)
        )
    )

  def __eq__(self, other: object) -> bool:
    if isinstance(other, PixelSpacing):
      return math.isclose(
          self.row_spacing_mm,
          other.row_spacing_mm,
          rel_tol=self._spacing_diff_tolerance,
      ) and math.isclose(
          self.column_spacing_mm,
          other.column_spacing_mm,
          rel_tol=self._spacing_diff_tolerance,
      )

    return False

  @property
  def row_spacing_mm(self) -> float:
    return self._row_spacing

  @property
  def column_spacing_mm(self) -> float:
    return self._column_spacing

  @property
  def pixel_spacing_mm(self) -> float:
    return self._pixel_spacing

  @property
  def as_magnification_string(self) -> str:
    """Returns a string of the zoom level corresponding to the pixel spacing."""
    magnification = self._scaling_factor / self._pixel_spacing
    if magnification.is_integer():
      return f'{int(magnification)}X'
    else:
      return f'{magnification}X'
