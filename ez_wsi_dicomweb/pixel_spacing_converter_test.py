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
"""Tests for pixel_spacing_converter."""
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import magnification as mag_lib
from ez_wsi_dicomweb import pixel_spacing_converter


class PixelSpacingConverterTest(parameterized.TestCase):

  def test_magnification_to_pixel_spacing(self):
    magnification = mag_lib.Magnification.FromString('5X')
    ps = pixel_spacing_converter.magnification_to_pixel_spacing(magnification)

    self.assertEqual(0.002, ps.pixel_spacing_mm)

  def test_magnification_string_to_pixel_spacing(self):
    ps = pixel_spacing_converter.magnification_string_to_pixel_spacing('5X')

    self.assertEqual(0.002, ps.pixel_spacing_mm)

  def test_magnification_to_pixel_spacing_raises(self):
    magnification_unknown = mag_lib.Magnification.Unknown()
    with self.assertRaises(ez_wsi_errors.MagnificationLevelNotFoundError):
      pixel_spacing_converter.magnification_to_pixel_spacing(
          magnification_unknown
      )


if __name__ == '__main__':
  absltest.main()
