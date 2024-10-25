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
"""Magnification tests."""

from typing import Union

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import magnification


class MagnificationTest(parameterized.TestCase):

  def test_Unknown(self):
    mag = magnification.Magnification.Unknown()
    self.assertEqual(
        mag._magnification_level,
        magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
    )

  @parameterized.parameters(
      ('40X', magnification.MagnificationLevel.M_40X),
      ('0.0390625X', magnification.MagnificationLevel.M_5X_DIV_128),
      ('unknown', magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
      ('30X', magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_FromString(
      self, str_value: str, expected_mag_level: magnification.MagnificationLevel
  ):
    self.assertEqual(
        expected_mag_level,
        magnification.Magnification.FromString(str_value)._magnification_level,
    )

  @parameterized.parameters(
      (40.0, magnification.MagnificationLevel.M_40X),
      (0.0390625, magnification.MagnificationLevel.M_5X_DIV_128),
      (-1, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
      (30.0, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_FromDouble(
      self,
      float_value: float,
      expected_mag_level: magnification.MagnificationLevel,
  ):
    self.assertEqual(
        expected_mag_level,
        magnification.Magnification.FromDouble(
            float_value
        )._magnification_level,
    )

  @parameterized.parameters(
      (0.25, magnification.MagnificationLevel.M_40X),
      (256, magnification.MagnificationLevel.M_5X_DIV_128),
      (0, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
      (0.19, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_FromPixelSize(
      self,
      pixel_size: float,
      expected_mag_level: magnification.MagnificationLevel,
  ):
    self.assertEqual(
        expected_mag_level,
        magnification.Magnification.FromPixelSize(
            pixel_size
        )._magnification_level,
    )

  @parameterized.parameters(
      (0.25, magnification.MagnificationLevel.M_40X),
      (256, magnification.MagnificationLevel.M_5X_DIV_128),
      (0, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_NominalPixelSize(
      self, pixel_size: float, mag_level: magnification.MagnificationLevel
  ):
    self.assertEqual(
        pixel_size, magnification.Magnification(mag_level).nominal_pixel_size
    )

  @parameterized.parameters(
      magnification.MagnificationLevel.M_40X,
      magnification.MagnificationLevel.M_5X_DIV_128,
      magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
  )
  def test_ToEnum(self, mag_level: magnification.MagnificationLevel):
    self.assertEqual(
        mag_level, magnification.Magnification(mag_level).magnification_level
    )

  @parameterized.parameters(
      (40.0, magnification.MagnificationLevel.M_40X),
      (0.0390625, magnification.MagnificationLevel.M_5X_DIV_128),
      (-1, magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_ToDouble(
      self, float_value: float, mag_level: magnification.MagnificationLevel
  ):
    self.assertEqual(
        float_value, magnification.Magnification(mag_level).as_double
    )

  @parameterized.parameters(
      ('40X', magnification.MagnificationLevel.M_40X),
      ('0.0390625X', magnification.MagnificationLevel.M_5X_DIV_128),
      ('unknown', magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION),
  )
  def test_ToString(
      self, str_value: str, mag_level: magnification.MagnificationLevel
  ):
    self.assertEqual(
        str_value, magnification.Magnification(mag_level).as_string
    )

  @parameterized.parameters(
      (magnification.MagnificationLevel.M_40X, False),
      (magnification.MagnificationLevel.M_5X_DIV_128, False),
      (magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION, True),
  )
  def test_IsUnknown(
      self, mag_level: magnification.MagnificationLevel, is_unknown: bool
  ):
    self.assertEqual(
        is_unknown, magnification.Magnification(mag_level).is_unknown
    )

  @parameterized.parameters(
      (
          magnification.MagnificationLevel.M_40X,
          magnification.MagnificationLevel.M_80X,
      ),
      (
          magnification.MagnificationLevel.M_5X_DIV_128,
          magnification.MagnificationLevel.M_5X_DIV_64,
      ),
      (
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
      ),
      (
          magnification.MagnificationLevel.M_100X,
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
      ),
  )
  def test_NextHigher(
      self,
      mag_level: magnification.MagnificationLevel,
      next_level: magnification.MagnificationLevel,
  ):
    self.assertEqual(
        next_level,
        magnification.Magnification(
            mag_level
        ).next_higher_magnification.magnification_level,
    )

  @parameterized.parameters(
      (
          magnification.MagnificationLevel.M_40X,
          magnification.MagnificationLevel.M_20X,
      ),
      (
          magnification.MagnificationLevel.M_5X_DIV_64,
          magnification.MagnificationLevel.M_5X_DIV_128,
      ),
      (
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
      ),
      (
          magnification.MagnificationLevel.M_5X_DIV_128,
          magnification.MagnificationLevel.UNKNOWN_MAGNIFICATION,
      ),
  )
  def test_next_lower_magnification_level(
      self,
      mag_level: magnification.MagnificationLevel,
      next_level: magnification.MagnificationLevel,
  ):
    self.assertEqual(
        next_level,
        magnification.Magnification(
            mag_level
        ).next_lower_magnification.magnification_level,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='equal',
          mag_1=magnification.Magnification.FromString('40X'),
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=True,
      ),
      dict(
          testcase_name='not_equal',
          mag_1=magnification.Magnification.FromString('20X'),
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=False,
      ),
      dict(
          testcase_name='unknown_from_string_not_equal',
          mag_1=magnification.Magnification.FromString('blah'),
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=False,
      ),
      dict(
          testcase_name='from_double_equal',
          mag_1=magnification.Magnification.FromDouble(40),
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=True,
      ),
      dict(
          testcase_name='from_nominal_pixel_size_equal',
          mag_1=magnification.Magnification.FromPixelSize(0.25),
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=True,
      ),
      dict(
          testcase_name='different_types_not_equal',
          mag_1='Potato',
          mag_2=magnification.Magnification.FromString('40X'),
          expected_result=False,
      ),
  )
  def test_magnfications_equal(
      self,
      mag_1: Union[magnification.Magnification, str],
      mag_2: magnification.Magnification,
      expected_result: bool,
  ):
    self.assertEqual(mag_1 == mag_2, expected_result)
    self.assertEqual(mag_2 == mag_1, expected_result)


if __name__ == '__main__':
  absltest.main()
