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
"""Functions for converting Magnification to PixelSpacing & vice versa.

These functions are separated from the PixelSpacing class so they can be more
easily removed once Magnification is fully removed.
"""
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import magnification as mag_lib
from ez_wsi_dicomweb import pixel_spacing


def magnification_to_pixel_spacing(
    magnification: mag_lib.Magnification,
) -> pixel_spacing.PixelSpacing:
  """Lightweight function that converts Magnification to PixelSpacing.

  Args:
    magnification: an instance of EZ WSI's magnification class to convert.

  Returns:
    A PixelSpacing instance created using the input magnification.

  Raises:
    MagnificationLevelNotFoundError if the provided magnification is unknown.
    InvalidMagnificationStringError if the string is not a non-zero positive
      floating point number.
  """
  if magnification.is_unknown:
    raise ez_wsi_errors.MagnificationLevelNotFoundError(
        'Provided magnification cannot be converted to PixelSpacing'
    )
  return pixel_spacing.PixelSpacing.FromMagnificationString(
      magnification.as_string
  )


def magnification_string_to_pixel_spacing(
    magnification: str,
) -> pixel_spacing.PixelSpacing:
  """Lightweight function that converts Magnification str to PixelSpacing."""
  return pixel_spacing.PixelSpacing.FromMagnificationString(magnification)
