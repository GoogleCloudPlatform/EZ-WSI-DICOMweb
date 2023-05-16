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
"""A class that represents a WSI slide magnification."""

from __future__ import annotations
import dataclasses
import enum
import math
from typing import Dict


@enum.unique
class MagnificationLevel(enum.Enum):
  """Enums of all supported magnification levels for DICOM/WSI slides."""

  UNKNOWN_MAGNIFICATION = 0
  M_100X = 1  # 100x
  M_80X = 2  # 80X
  M_40X = 3  # 40x
  M_20X = 4  # 20x
  M_10X = 5  # 10x
  M_5X = 6  # 5x
  M_5X_DIV_2 = 7  # 2.5x = 5x / 2
  M_5X_DIV_4 = 8  # 1.25x = 5x / 4
  M_5X_DIV_8 = 9  # 0.625x = 5x / 8
  M_5X_DIV_16 = 10  # 0.3125x = 5x / 16
  M_5X_DIV_32 = 11  # 0.15625 = 5x / 32
  M_5X_DIV_64 = 12  # 0.078125 = 5x / 64
  M_5X_DIV_128 = 13  # 0.0390625 = 5x / 128


@dataclasses.dataclass(frozen=True)
class MagnificationProperties:
  """Class for storing all properties associated with a magnification.

  Used to build a fast lookup table in
  _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES.

  Attributes:
    as_double: The floating point value of the magnification.
    as_string: The string label of the magnification.
    nominal_pixel_size: The nominal pixel size in mpp (micrometers per pixel).
  """

  as_double: float
  as_string: str
  nominal_pixel_size: float


# A dict that maps all magnification levels to their properties.
_MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES: Dict[
    MagnificationLevel, MagnificationProperties
] = {
    MagnificationLevel.UNKNOWN_MAGNIFICATION: MagnificationProperties(
        as_double=-1.0, as_string='unknown', nominal_pixel_size=0
    ),
    MagnificationLevel.M_100X: MagnificationProperties(
        as_double=100.0, as_string='100X', nominal_pixel_size=0.1
    ),
    MagnificationLevel.M_80X: MagnificationProperties(
        as_double=80.0, as_string='80X', nominal_pixel_size=0.125
    ),
    MagnificationLevel.M_40X: MagnificationProperties(
        as_double=40.0, as_string='40X', nominal_pixel_size=0.25
    ),
    MagnificationLevel.M_20X: MagnificationProperties(
        as_double=20.0, as_string='20X', nominal_pixel_size=0.5
    ),
    MagnificationLevel.M_10X: MagnificationProperties(
        as_double=10.0, as_string='10X', nominal_pixel_size=1.0
    ),
    MagnificationLevel.M_5X: MagnificationProperties(
        as_double=5.0, as_string='5X', nominal_pixel_size=2.0
    ),
    MagnificationLevel.M_5X_DIV_2: MagnificationProperties(
        as_double=2.5, as_string='2.5X', nominal_pixel_size=4.0
    ),
    MagnificationLevel.M_5X_DIV_4: MagnificationProperties(
        as_double=1.25, as_string='1.25X', nominal_pixel_size=8.0
    ),
    MagnificationLevel.M_5X_DIV_8: MagnificationProperties(
        as_double=0.625, as_string='0.625X', nominal_pixel_size=16
    ),
    MagnificationLevel.M_5X_DIV_16: MagnificationProperties(
        as_double=0.3125, as_string='0.3125X', nominal_pixel_size=32.0
    ),
    MagnificationLevel.M_5X_DIV_32: MagnificationProperties(
        as_double=0.15625, as_string='0.15625X', nominal_pixel_size=64.0
    ),
    MagnificationLevel.M_5X_DIV_64: MagnificationProperties(
        as_double=0.078125, as_string='0.078125X', nominal_pixel_size=128.0
    ),
    MagnificationLevel.M_5X_DIV_128: MagnificationProperties(
        as_double=0.0390625, as_string='0.0390625X', nominal_pixel_size=256.0
    ),
}

# The tolerance for matching a un-normalized pixel size value to a
# magnification level.
_MPP_SCALE_TOLERANCE = 0.2


class Magnification:
  """Represents a magnification level of a DICOM/WSI slide."""

  def __init__(self, magnification_level: MagnificationLevel):
    """Constructor.

    Args:
      magnification_level: The magnification.
    """
    self._magnification_level = magnification_level

  @classmethod
  def Unknown(cls):
    """Returns a Magnification object at UNKNOWN_MAGNIFICATION."""
    return Magnification(MagnificationLevel.UNKNOWN_MAGNIFICATION)

  @classmethod
  def FromString(cls, mag_str: str):
    """Returns a Magnification object given a string potentially representing a magnification.

    Args:
      mag_str: The string value of the magnfication to search for.
    """
    for (
        level,
        mag_level_props,
    ) in _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES.items():
      if mag_str == mag_level_props.as_string:
        return Magnification(level)
    return cls.Unknown()

  @classmethod
  def FromDouble(cls, mag_float: float):
    """Returns a Magnification by its floating value representation.

    Args:
      mag_float: The floating value of the magnfication to request.
    """
    for (
        mag_level,
        mag_level_props,
    ) in _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES.items():
      if math.isclose(mag_float, mag_level_props.as_double):
        return Magnification(mag_level)
    return cls.Unknown()

  @classmethod
  def FromPixelSize(cls, mpp: float):
    """Returns a Magnification by a pixel size.

    Args:
      mpp: The pixel size, in micrometers per pixel, of the magnification to
        request.
    """
    for (
        mag_level,
        mag_level_props,
    ) in _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES.items():
      if mag_level == MagnificationLevel.UNKNOWN_MAGNIFICATION:
        continue
      scale = mpp / mag_level_props.nominal_pixel_size
      if abs(scale - 1) < _MPP_SCALE_TOLERANCE:
        return Magnification(mag_level)
    return cls.Unknown()

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Magnification):
      return self._magnification_level == other._magnification_level

    return False

  @property
  def nominal_pixel_size(self) -> float:
    """Returns the nominal pixel size of the magnification."""
    return _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES[
        self._magnification_level
    ].nominal_pixel_size

  @property
  def magnification_level(self) -> MagnificationLevel:
    """Returns the enum value of the magnification level."""
    return self._magnification_level

  @property
  def as_double(self) -> float:
    """Returns the floating value of the magnification level."""
    return _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES[
        self._magnification_level
    ].as_double

  @property
  def as_string(self) -> str:
    """Returns the string value of the magnification level."""
    return _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES[
        self._magnification_level
    ].as_string

  @property
  def is_unknown(self) -> bool:
    """Checks if the magnification is unknown."""
    return self._magnification_level == MagnificationLevel.UNKNOWN_MAGNIFICATION

  @property
  def next_higher_magnification(self) -> Magnification:
    """Creates a Magnification object containing next higher magnification level.

    Returns:
      An object with next higher magnification, or UNKNOWN magnification if this
      is the highest magnification level.
    """
    return self.FromPixelSize(
        _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES[
            self._magnification_level
        ].nominal_pixel_size
        / 2
    )

  @property
  def next_lower_magnification(self) -> Magnification:
    """Creates a Magnification object containing next lower magnification level.

    Returns:
      An object with the next lower magnification, or UNKNOWN magnification if
      this is the lowest magnification level.
    """
    return self.FromPixelSize(
        _MAGNIFICATION_LEVEL_TO_MAGNIFICATION_PROPERTIES[
            self._magnification_level
        ].nominal_pixel_size
        * 2
    )
