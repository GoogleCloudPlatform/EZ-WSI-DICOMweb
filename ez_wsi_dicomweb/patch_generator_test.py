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
"""Tests for patch_generator."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import patch_generator as patch_generator_lib
from ez_wsi_dicomweb import pixel_spacing
from hcls_imaging_ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
from PIL import Image


class PatchGeneratorTest(parameterized.TestCase):

  def test_iterate_tissue_mask(self):
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    expected = np.array(
        Image.open(dicom_test_utils.testdata_path('golden_inference_mask.png'))
    )
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
    )
    np.testing.assert_array_almost_equal(
        expected,
        patch_generator._normalized_tissue_mask().astype(np.uint8) * 255,
    )

    patch1 = mock.create_autospec(dicom_slide.Patch)
    patch2 = mock.create_autospec(dicom_slide.Patch)
    mock_slide.get_patch.side_effect = [patch1, patch2]
    it = iter(patch_generator)
    self.assertEqual(patch1, next(it))
    self.assertEqual(patch2, next(it))

  def test_total_num_patches(self):
    # 4x8 is white and should be filtered; 8x8 is tissue and should be kept
    patch_arr = np.concatenate(
        (np.ones((4, 8, 3)), np.ones((8, 8, 3)) * 0.1), axis=0
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
    )

    # stride size 128 at 40X is 4 at 1.25X. Expected number of tissue patches is
    # 8 * 8 / 4 / 4 = 4
    self.assertEqual(4, patch_generator.total_num_patches())

  def test_total_num_patches_custom_magnification(self):
    # 4x8 is white and should be filtered; 8x8 is tissue and should be kept
    patch_arr = np.concatenate(
        (np.ones((4, 8, 3)), np.ones((8, 8, 3)) * 0.1), axis=0
    )

    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
        tissue_mask_pixel_spacing=pixel_spacing.PixelSpacing.FromMagnificationString(
            '2.5X'
        ),
    )

    # stride size 128 at 40X is 8 at 2.5X. Expected number of tissue patches is
    # 8 * 8 / 8 / 8 = 1
    self.assertEqual(1, patch_generator.total_num_patches())

  def test_total_strides(self):
    patch_arr = np.ones((8, 8, 3)) * 0.1
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
    )

    # stride size 128 at 40X is 4 at 1.25X. Expected y_strides is 8 / 4 = 2,
    # expected x_strides is 8 / 4 = 2.
    self.assertEqual(
        patch_generator_lib.StrideCoordinate(y_strides=2, x_strides=2),
        patch_generator.total_strides(),
    )

  def test_total_strides_large_stride_size(self):
    patch_arr = np.ones((8, 8, 3)) * 0.1
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=512,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
    )

    # stride size 512 at 40X is 16 at 1.25X. Expected y_strides is
    # max(8 / 16, 1) = 1, expected x_strides is max(8 / 16, 1) = 1.
    self.assertEqual(
        patch_generator_lib.StrideCoordinate(y_strides=1, x_strides=1),
        patch_generator.total_strides(),
    )

  def test_empty_tissue_mask(self):
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.Image)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    slide_uid = 'slide_uid'
    series_uid = 'series_uid'
    dicom_full_path = (
        'projects/project_name/locations/us-west1/datasets/'
        'dataset_name/dicomStores/dicom_store_name/dicomWeb/'
        f'studies/{slide_uid}/series/{series_uid}'
    )
    mock_slide.path = dicom_path.FromString(dicom_full_path)

    threshold = 0
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=threshold,
    )
    with self.assertRaisesRegex(
        ez_wsi_errors.DicomImageMissingRegionError,
        (
            f'Slide with study_uid {slide_uid} series_uid {series_uid} has no '
            f'region with luminance value less than threshold {threshold}.'
        ),
    ):
      next(iter(patch_generator))

  @parameterized.named_parameters(
      dict(
          testcase_name='base_case',
          y_strides=0,
          x_strides=0,
          patch_size=128,
          stride_size=128,
          expected_patch_bounds=dicom_slide.PatchBounds(
              x_origin=0, y_origin=0, width=128, height=128
          ),
      ),
      # Given that patches are originiated in the upper left corner a larger
      # patch should be offset to the left.
      dict(
          testcase_name='larger_patch',
          y_strides=0,
          x_strides=0,
          patch_size=299,
          stride_size=128,
          expected_patch_bounds=dicom_slide.PatchBounds(
              x_origin=-85, y_origin=-85, width=299, height=299
          ),
      ),
      # Increasing stride should shift patch right, down by (stride-patch)/2
      dict(
          testcase_name='larger_stride',
          y_strides=0,
          x_strides=0,
          patch_size=128,
          stride_size=299,
          expected_patch_bounds=dicom_slide.PatchBounds(
              x_origin=85, y_origin=85, width=128, height=128
          ),
      ),
      # Moving coordinates should shift by stride
      dict(
          testcase_name='offset_y_stride',
          y_strides=1,
          x_strides=0,
          patch_size=299,
          stride_size=128,
          expected_patch_bounds=dicom_slide.PatchBounds(
              x_origin=-85, y_origin=(-85 + 128), width=299, height=299
          ),
      ),
      # Moving coordinates should shift by stride
      dict(
          testcase_name='offset_x_stride',
          y_strides=0,
          x_strides=1,
          patch_size=299,
          stride_size=128,
          expected_patch_bounds=dicom_slide.PatchBounds(
              x_origin=(-85 + 128), y_origin=-85, width=299, height=299
          ),
      ),
  )
  def test_coordinates_conversion(
      self, y_strides, x_strides, patch_size, stride_size, expected_patch_bounds
  ):
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    patch_generator = patch_generator_lib.PatchGenerator(
        mock_slide,
        pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
        stride_size,
        patch_size,
    )
    self.assertEqual(
        expected_patch_bounds,
        patch_generator.strides_to_patch_bounds(
            patch_generator_lib.StrideCoordinate(y_strides, x_strides)
        ),
    )
    self.assertEqual(
        patch_generator_lib.StrideCoordinate(y_strides, x_strides),
        patch_generator.patch_bounds_to_strides(expected_patch_bounds),
    )


if __name__ == '__main__':
  absltest.main()
