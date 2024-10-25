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
"""Tests for pixel_spacing."""
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import pixel_spacing


class PixelSpacingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='1_percent_different_pixels',
          row_spacing=100,
          column_spacing=99,
          expected_pixel_spacing=99,
          expected_row_spacing=100,
          expected_column_spacing=99,
      ),
      dict(
          testcase_name='1_percent_different_pixels_inverted',
          row_spacing=99,
          column_spacing=100,
          expected_pixel_spacing=99,
          expected_row_spacing=99,
          expected_column_spacing=100,
      ),
      dict(
          testcase_name='square_pixels',
          row_spacing=0.01,
          column_spacing=0.01,
          expected_pixel_spacing=0.01,
          expected_row_spacing=0.01,
          expected_column_spacing=0.01,
      ),
      dict(
          testcase_name='small_square_pixels',
          row_spacing=0.000001,
          column_spacing=0.0000010001,
          expected_pixel_spacing=0.00000100005,
          expected_row_spacing=0.000001,
          expected_column_spacing=0.0000010001,
      ),
  )
  def test_square_pixels(
      self,
      row_spacing: float,
      column_spacing: float,
      expected_pixel_spacing: float,
      expected_row_spacing: float,
      expected_column_spacing: float,
  ):
    ps = pixel_spacing.PixelSpacing(column_spacing, row_spacing)

    self.assertAlmostEqual(expected_pixel_spacing, ps.pixel_spacing_mm)
    self.assertAlmostEqual(expected_row_spacing, ps.row_spacing_mm)
    self.assertAlmostEqual(expected_column_spacing, ps.column_spacing_mm)

  @parameterized.named_parameters(
      dict(
          testcase_name='large_pixel_spacing',
          pixel_spacing_str='100.5',
          expected_pixel_spacing=100.5,
      ),
      dict(
          testcase_name='small_pixel_spacing',
          pixel_spacing_str='0.00000100005',
          expected_pixel_spacing=0.00000100005,
      ),
  )
  def test_from_string(
      self,
      pixel_spacing_str: str,
      expected_pixel_spacing: float,
  ):
    ps = pixel_spacing.PixelSpacing.FromString(pixel_spacing_str)

    self.assertEqual(expected_pixel_spacing, ps.pixel_spacing_mm)

  @parameterized.named_parameters(
      dict(
          testcase_name='5x_zoom',
          magnification_str='5x',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.002, 0.002),
          equal=True,
      ),
      dict(
          testcase_name='5X_zoom',
          magnification_str='5X',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.002, 0.002),
          equal=True,
      ),
      dict(
          testcase_name='5_zoom',
          magnification_str='5',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.002, 0.002),
          equal=True,
      ),
      dict(
          testcase_name='2.5X_zoom',
          magnification_str='2.5X',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.004, 0.004),
          equal=True,
      ),
      dict(
          testcase_name='2.5_zoom',
          magnification_str='2.5',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.004, 0.004),
          equal=True,
      ),
      dict(
          testcase_name='200_zoom',
          magnification_str='200',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.00005, 0.00005),
          equal=True,
      ),
      dict(
          testcase_name='5.1_zoom',
          magnification_str='5.1',
          expected_pixel_spacing=pixel_spacing.PixelSpacing(0.003, 0.003),
          equal=False,
      ),
  )
  def test_from_magnification_string(
      self,
      magnification_str: str,
      expected_pixel_spacing: pixel_spacing.PixelSpacing,
      equal: bool,
  ):
    ps = pixel_spacing.PixelSpacing.FromMagnificationString(magnification_str)
    self.assertEqual(
        equal,
        expected_pixel_spacing == ps,
        (
            ps.column_spacing_mm,
            ps.row_spacing_mm,
            expected_pixel_spacing.column_spacing_mm,
            expected_pixel_spacing.row_spacing_mm,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='all_dots',
          magnification_str='....',
      ),
      dict(
          testcase_name='one_dot',
          magnification_str='.',
      ),
      dict(
          testcase_name='weird_dots',
          magnification_str='.X5.',
      ),
      dict(
          testcase_name='wrong_order',
          magnification_str='X5',
      ),
      dict(
          testcase_name='negative',
          magnification_str='-5',
      ),
      dict(
          testcase_name='zero',
          magnification_str='0.0',
      ),
  )
  def test_from_magnification_string_error(
      self,
      magnification_str: str,
  ):
    with self.assertRaises(ez_wsi_errors.InvalidMagnificationStringError):
      pixel_spacing.PixelSpacing.FromMagnificationString(magnification_str)

  @parameterized.named_parameters(
      dict(
          testcase_name='large_pixel_spacing',
          float_pixel_spacing=100.5,
          expected_pixel_spacing=100.5,
      ),
      dict(
          testcase_name='small_pixel_spacing',
          float_pixel_spacing=0.00000100005,
          expected_pixel_spacing=0.00000100005,
      ),
  )
  def test_from_double(
      self,
      float_pixel_spacing: float,
      expected_pixel_spacing: float,
  ):
    ps = pixel_spacing.PixelSpacing.FromDouble(float_pixel_spacing)

    self.assertEqual(expected_pixel_spacing, ps.pixel_spacing_mm)

  @parameterized.named_parameters(
      dict(
          testcase_name='large_pixel_spacings_equal',
          first_pixel_spacing=100.5,
          second_pixel_spacing=100.5,
      ),
      dict(
          testcase_name='small_pixel_spacings_equal',
          first_pixel_spacing=0.00000100005,
          second_pixel_spacing=0.00000100005,
      ),
      dict(
          testcase_name='small_pixel_spacings_roughly_equal',
          first_pixel_spacing=0.00000100005,
          second_pixel_spacing=0.0000010000505,
      ),
  )
  def test_from_double_equal(
      self,
      first_pixel_spacing: float,
      second_pixel_spacing: float,
  ):
    ps_one = pixel_spacing.PixelSpacing.FromDouble(first_pixel_spacing)
    ps_two = pixel_spacing.PixelSpacing.FromDouble(second_pixel_spacing)

    self.assertEqual(ps_one, ps_two)

  @parameterized.named_parameters(
      dict(
          testcase_name='large_pixel_spacings_not_equal',
          first_pixel_spacing=100.5,
          second_pixel_spacing=106.0,
      ),
      dict(
          testcase_name='small_pixel_spacings_not_equal',
          first_pixel_spacing=0.000001100005,
          second_pixel_spacing=0.00000100005,
      ),
  )
  def test_from_double_not_equal(
      self, first_pixel_spacing: float, second_pixel_spacing: float
  ):
    ps_one = pixel_spacing.PixelSpacing.FromDouble(first_pixel_spacing)
    ps_two = pixel_spacing.PixelSpacing.FromDouble(second_pixel_spacing)

    self.assertNotEqual(ps_one, ps_two)

  @parameterized.named_parameters(
      dict(
          testcase_name='hashes_equal',
          first_row_spacing=100,
          first_column_spacing=99,
          second_row_spacing=100,
          second_column_spacing=99,
          expected_result=True,
      ),
      dict(
          testcase_name='hashes_not_equal',
          first_row_spacing=100,
          first_column_spacing=99,
          second_row_spacing=99,
          second_column_spacing=100,
          expected_result=False,
      ),
      dict(
          testcase_name='hashes_really_not_equal',
          first_row_spacing=100,
          first_column_spacing=99,
          second_row_spacing=99,
          second_column_spacing=99,
          expected_result=False,
      ),
  )
  def test_hashes_equal(
      self,
      first_row_spacing: float,
      first_column_spacing: float,
      second_row_spacing: float,
      second_column_spacing: float,
      expected_result: bool,
  ):
    ps_one = pixel_spacing.PixelSpacing(first_column_spacing, first_row_spacing)
    ps_two = pixel_spacing.PixelSpacing(
        second_column_spacing, second_row_spacing
    )

    self.assertEqual(ps_one.__hash__() == ps_two.__hash__(), expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='string_not_equal',
          first_pixel_spacing=100.5,
          second_object='hello',
      ),
      dict(
          testcase_name='double_not_equal',
          first_pixel_spacing=100.5,
          second_object=100.5,
      ),
  )
  def test_from_double_equal_not_ps(
      self,
      first_pixel_spacing: float,
      second_object: object,
  ):
    ps_one = pixel_spacing.PixelSpacing.FromDouble(first_pixel_spacing)

    self.assertFalse(ps_one.__eq__(second_object))

  @parameterized.named_parameters(
      dict(
          testcase_name='large_zoom',
          float_pixel_spacing=0.0001,
          expected_magnification_string='100X',
      ),
      dict(
          testcase_name='80x_zoom',
          float_pixel_spacing=0.000125,
          expected_magnification_string='80X',
      ),
      dict(
          testcase_name='40x_zoom',
          float_pixel_spacing=0.00025,
          expected_magnification_string='40X',
      ),
      dict(
          testcase_name='20x_zoom',
          float_pixel_spacing=0.0005,
          expected_magnification_string='20X',
      ),
      dict(
          testcase_name='10x_zoom',
          float_pixel_spacing=0.001,
          expected_magnification_string='10X',
      ),
      dict(
          testcase_name='5x_zoom',
          float_pixel_spacing=0.002,
          expected_magnification_string='5X',
      ),
      dict(
          testcase_name='2.5x_zoom',
          float_pixel_spacing=0.004,
          expected_magnification_string='2.5X',
      ),
      dict(
          testcase_name='1.25x_zoom',
          float_pixel_spacing=0.008,
          expected_magnification_string='1.25X',
      ),
      dict(
          testcase_name='.625x_zoom',
          float_pixel_spacing=0.016,
          expected_magnification_string='0.625X',
      ),
      dict(
          testcase_name='.3125x_zoom',
          float_pixel_spacing=0.032,
          expected_magnification_string='0.3125X',
      ),
      dict(
          testcase_name='.15625x_zoom',
          float_pixel_spacing=0.064,
          expected_magnification_string='0.15625X',
      ),
      dict(
          testcase_name='.078125x_zoom',
          float_pixel_spacing=0.128,
          expected_magnification_string='0.078125X',
      ),
      dict(
          testcase_name='smallest_zoom',
          float_pixel_spacing=0.256,
          expected_magnification_string='0.0390625X',
      ),
  )
  def test_as_magnification_string(
      self,
      float_pixel_spacing: float,
      expected_magnification_string: str,
  ):
    ps = pixel_spacing.PixelSpacing.FromDouble(float_pixel_spacing)

    self.assertEqual(expected_magnification_string, ps.as_magnification_string)

  def test_pixel_spacing_consistency_from_double(self):
    # Test string -> PS is equal to string -> PS -> double -> PS
    self.assertEqual(
        pixel_spacing.PixelSpacing.FromMagnificationString('10X'),
        pixel_spacing.PixelSpacing.FromDouble(
            pixel_spacing.PixelSpacing.FromMagnificationString(
                '10X'
            ).pixel_spacing_mm
        ),
    )

  def test_pixel_spacing_consistency_from_string(self):
    # Test string -> PS is equal to string -> PS -> string -> PS
    self.assertEqual(
        pixel_spacing.PixelSpacing.FromMagnificationString('10X'),
        pixel_spacing.PixelSpacing.FromMagnificationString(
            pixel_spacing.PixelSpacing.FromMagnificationString(
                '10X'
            ).as_magnification_string
        ),
    )

  def test_scaling_factor(self):
    self.assertEqual(
        pixel_spacing.PixelSpacing(0.01, 0.01).as_magnification_string,
        pixel_spacing.PixelSpacing(0.02, 0.02, 0.02).as_magnification_string,
    )

  def test_scaling_factor_from_string(self):
    self.assertEqual(
        pixel_spacing.PixelSpacing.FromString('.01').as_magnification_string,
        pixel_spacing.PixelSpacing.FromString(
            '.02', 0.02
        ).as_magnification_string,
    )

  def test_string_scaling_factor_from_magnification(self):
    self.assertEqual(
        pixel_spacing.PixelSpacing.FromMagnificationString(
            '5X'
        ).as_magnification_string,
        pixel_spacing.PixelSpacing.FromMagnificationString(
            '5X', 0.02
        ).as_magnification_string,
    )

  def test_scaling_factor_from_double(self):
    self.assertEqual(
        pixel_spacing.PixelSpacing.FromDouble(0.01).as_magnification_string,
        pixel_spacing.PixelSpacing.FromDouble(
            0.02, 0.02
        ).as_magnification_string,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='10_percent_different_pixels',
          row_spacing=100,
          column_spacing=90,
          diff_tolerance=0.1,
          expected_pixel_spacing=90,
          expected_row_spacing=100,
          expected_column_spacing=90,
      ),
      dict(
          testcase_name='10_percent_different_pixels_inverted',
          row_spacing=90,
          column_spacing=100,
          diff_tolerance=0.1,
          expected_pixel_spacing=90,
          expected_row_spacing=90,
          expected_column_spacing=100,
      ),
  )
  def test_square_pixels_with_diff_tolerance(
      self,
      row_spacing: float,
      column_spacing: float,
      diff_tolerance: float,
      expected_pixel_spacing: float,
      expected_row_spacing: float,
      expected_column_spacing: float,
  ):
    ps = pixel_spacing.PixelSpacing(
        column_spacing, row_spacing, spacing_diff_tolerance=diff_tolerance
    )

    self.assertAlmostEqual(expected_pixel_spacing, ps.pixel_spacing_mm)
    self.assertAlmostEqual(expected_row_spacing, ps.row_spacing_mm)
    self.assertAlmostEqual(expected_column_spacing, ps.column_spacing_mm)

  def test_undefined_pixel_spacing_is_not_defined(self):
    self.assertFalse(pixel_spacing.UndefinedPixelSpacing().is_defined)

  def test_undefined_pixel_spacing_raises_if_spacing_attributes_accessed(self):
    with self.assertRaises(ez_wsi_errors.UndefinedPixelSpacingError):
      _ = pixel_spacing.UndefinedPixelSpacing().column_spacing_mm
    with self.assertRaises(ez_wsi_errors.UndefinedPixelSpacingError):
      _ = pixel_spacing.UndefinedPixelSpacing().row_spacing_mm
    with self.assertRaises(ez_wsi_errors.UndefinedPixelSpacingError):
      _ = pixel_spacing.UndefinedPixelSpacing().pixel_spacing_mm

  @parameterized.named_parameters([
      dict(
          testcase_name='equal_pixel_spacing',
          ps1=pixel_spacing.PixelSpacing(0.01, 0.01),
          ps2=pixel_spacing.PixelSpacing(0.01, 0.01),
          expected=True,
      ),
      dict(
          testcase_name='not_equal_pixel_spacing_1',
          ps1=pixel_spacing.PixelSpacing(0.01, 0.01),
          ps2=pixel_spacing.PixelSpacing(0.02, 0.01),
          expected=False,
      ),
      dict(
          testcase_name='not_equal_pixel_spacing_2',
          ps1=pixel_spacing.PixelSpacing(0.01, 0.01),
          ps2=pixel_spacing.PixelSpacing(0.01, 0.02),
          expected=False,
      ),
      dict(
          testcase_name='not_equal_pixel_spacing_3',
          ps1=pixel_spacing.PixelSpacing(0.01, 0.01),
          ps2=pixel_spacing.PixelSpacing(0.02, 0.02),
          expected=False,
      ),
      dict(
          testcase_name='undefined_and_defined',
          ps1=pixel_spacing.PixelSpacing(0.01, 0.01),
          ps2=pixel_spacing.UndefinedPixelSpacing(),
          expected=False,
      ),
      dict(
          testcase_name='undefined_and_undefined',
          ps1=pixel_spacing.UndefinedPixelSpacing(),
          ps2=pixel_spacing.UndefinedPixelSpacing(),
          expected=True,
      ),
  ])
  def test_pixel_spacing_equality(self, ps1, ps2, expected):
    self.assertEqual(ps1 == ps2, expected)


if __name__ == '__main__':
  absltest.main()
