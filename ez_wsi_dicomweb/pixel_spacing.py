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

import dataclasses
import math
from typing import Any, List, Optional

import dataclasses_json
from ez_wsi_dicomweb import ez_wsi_errors


# The tolerance (percentage difference) for difference between pixel spacings.
# EZ WSI expects practically square pixels.
PIXEL_SPACING_DIFF_TOLERANCE = 0.05
# Magnification & Pixel Spacing are usually linearly related via a constant
# scaling factor of 0.01. he imaging scale factor can be directly derived via a
# single picture of a microscope calibration slide captured at a known
# magnification.
_SCALE_FACTOR = 0.01


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class PixelSpacing:
  """Represents the pixel spacing of a DICOM/WSI slide."""

  # Do not access _column_spacing_mm directly.
  # Instead use column_spacing_mm property, property raises if accessing
  # undefined (None) _column_spacing_mm.
  _column_spacing_mm: Optional[float] = dataclasses.field(
      metadata=dataclasses_json.config(field_name='column_spacing_mm')
  )

  # Do not access _row_spacing_mm directly.
  # Instead use row_spacing_mm property, property raises if accessing
  # undefined (None) _row_spacing_mm.
  _row_spacing_mm: Optional[float] = dataclasses.field(
      metadata=dataclasses_json.config(field_name='row_spacing_mm')
  )
  mag_scaling_factor: float = _SCALE_FACTOR
  spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE

  @property
  def column_spacing_mm(self) -> float:
    """Returns column pixel spacing (mm/pixel).

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: Pixel spacing not defined.
    """
    if self._column_spacing_mm is None:
      raise ez_wsi_errors.UndefinedPixelSpacingError()
    return self._column_spacing_mm

  @property
  def row_spacing_mm(self) -> float:
    """Returns row pixel spacing (mm/pixel).

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: Pixel spacing not defined.
    """
    if self._row_spacing_mm is None:
      raise ez_wsi_errors.UndefinedPixelSpacingError()
    return self._row_spacing_mm

  @property
  def is_defined(self) -> bool:
    return (
        self._column_spacing_mm is not None and self._row_spacing_mm is not None
    )

  @classmethod
  def FromDicomPixelSpacingTag(
      cls, pixel_spacing_tag: List[float]
  ) -> PixelSpacing:
    # DICOM Pixel spacing tag is row, column ordered.
    # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_10.7.html#sect_10.7.1.3
    row_spacing, column_spacing = pixel_spacing_tag
    return PixelSpacing(column_spacing, row_spacing)

  @classmethod
  def FromString(
      cls,
      pixel_spacing: str,
      scaling_factor: float = _SCALE_FACTOR,
      spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE,
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a string representing a pixel spacing measured in mm/px creates a
    PixelSpacing object.

    Args:
      pixel_spacing: The string value of the pixel spacing, e.g. a string
        representing a double measured in mm/px.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.
      spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between pixel spacings.

    Returns:
      A new PixelSpacing object.
    """
    return PixelSpacing.FromDouble(
        pixel_spacing=float(pixel_spacing),
        scaling_factor=scaling_factor,
        spacing_diff_tolerance=spacing_diff_tolerance,
    )

  @classmethod
  def FromMagnificationString(
      cls,
      magnification: str,
      scaling_factor: float = _SCALE_FACTOR,
      spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE,
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a string representing a Magnificiation level creates a PixelSpacing
    object.

    Args:
      magnification: The string value of the magnification, e.g. a string
        representing a zoom level: 5, 5X, 10, 10X, 25X.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.
      spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between pixel spacings.

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
        pixel_spacing=pixel_spacing,
        scaling_factor=scaling_factor,
        spacing_diff_tolerance=spacing_diff_tolerance,
    )

  @classmethod
  def FromDouble(
      cls,
      pixel_spacing: float,
      scaling_factor: float = _SCALE_FACTOR,
      spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE,
  ) -> PixelSpacing:
    """Returns a PixelSpacing object.

    Given a pixel spacing float measured in mm/px creates a PixelSpacing object.

    Args:
      pixel_spacing: The avg value of the row & column pixel spacing in mm/px.
      scaling_factor: a number representing the scaling factor between a pixel
        spacing and a zoom level or magnification.
      spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between pixel spacings.

    Returns:
      A new PixelSpacing object.
    """
    return PixelSpacing(
        _column_spacing_mm=pixel_spacing,
        _row_spacing_mm=pixel_spacing,
        mag_scaling_factor=scaling_factor,
        spacing_diff_tolerance=spacing_diff_tolerance,
    )

  def __hash__(self):
    return hash(dataclasses.astuple(self))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, PixelSpacing):
      return False
    try:
      return math.isclose(
          self.row_spacing_mm,
          other.row_spacing_mm,
          rel_tol=self.spacing_diff_tolerance,
      ) and math.isclose(
          self.column_spacing_mm,
          other.column_spacing_mm,
          rel_tol=self.spacing_diff_tolerance,
      )
    except ez_wsi_errors.UndefinedPixelSpacingError:
      return self.is_defined == other.is_defined

  @property
  def pixel_spacing_mm(self) -> float:
    """Returns smallest column/row pixel spacing (mm/pixel).

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: Pixel spacing not defined.
    """
    return min(self.row_spacing_mm, self.column_spacing_mm)

  @property
  def as_magnification_string(self) -> str:
    """Returns a string of the zoom level corresponding to the pixel spacing.

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: Pixel spacing not defined.
    """
    magnification = self.mag_scaling_factor / self.pixel_spacing_mm
    if magnification.is_integer():
      return f'{int(magnification)}X'
    else:
      return f'{magnification}X'


def UndefinedPixelSpacing(
    mag_scaling_factor: float = _SCALE_FACTOR,
    spacing_diff_tolerance: float = PIXEL_SPACING_DIFF_TOLERANCE,
) -> PixelSpacing:
  return PixelSpacing(
      _column_spacing_mm=None,
      _row_spacing_mm=None,
      mag_scaling_factor=mag_scaling_factor,
      spacing_diff_tolerance=spacing_diff_tolerance,
  )
