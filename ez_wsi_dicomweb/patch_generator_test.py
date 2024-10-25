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
"""Tests for patch generator."""

import os
import shutil
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_generator as patch_generator_lib
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
from PIL import Image
import pydicom

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


def _get_mock_slide_level(
    ps: pixel_spacing.PixelSpacing, tissue_mask: np.ndarray, scale_factor: float
) -> slide_level_map.Level:
  mk_level = mock.create_autospec(slide_level_map.Level, instance=True)
  type(mk_level).pixel_spacing = mock.PropertyMock(return_value=ps)
  mk_level.width = int(tissue_mask.shape[1] * scale_factor)
  mk_level.height = int(tissue_mask.shape[0] * scale_factor)
  # disable preloading tissue mask not supported by test.
  max_frame_number = (
      patch_generator_lib._MAX_TISSUE_MASK_LEVEL_FRAME_COUNT_PRELOAD + 1
  )
  mk_level.frame_number_max = max_frame_number
  type(mk_level).number_of_frames = mock.PropertyMock(
      return_value=max_frame_number
  )
  return mk_level


class PatchGeneratorTest(parameterized.TestCase):

  def test_iterate_tissue_mask(self):
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    expected = np.array(
        Image.open(dicom_test_utils.testdata_path('golden_inference_mask.png'))
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
    )
    np.testing.assert_array_almost_equal(
        expected,
        patch_generator.normalized_tissue_mask.astype(np.uint8) * 255,
    )

    patch1 = mock.create_autospec(dicom_slide.DicomPatch)
    patch2 = mock.create_autospec(dicom_slide.DicomPatch)
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
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
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
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 2.5,
        ),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=0.8,
        mask_level=mock_slide.get_level_by_pixel_spacing(
            pixel_spacing.PixelSpacing.FromMagnificationString('2.5X')
        ),
    )

    # stride size 128 at 40X is 8 at 2.5X. Expected number of tissue patches is
    # 8 * 8 / 8 / 8 = 1
    self.assertEqual(1, patch_generator.total_num_patches())

  def test_total_strides(self):
    patch_arr = np.ones((8, 8, 3)) * 0.1
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
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
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
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
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
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
    mock_slide.get_level_by_pixel_spacing.return_value = _get_mock_slide_level(
        pixel_spacing.PixelSpacing.FromMagnificationString('1.25'),
        patch_arr,
        1.0,
    )
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
        stride_size=128,
        patch_size=patch_arr.shape[0],
        max_luminance=threshold,
    )
    with self.assertRaisesRegex(
        ez_wsi_errors.DicomImageMissingRegionError,
        (
            'Tissue mask has no regions with luminance value within threshold'
            ' 0.00392156862745098 - 0.'
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
              x_origin=0, y_origin=0, width=299, height=299
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
              x_origin=-0, y_origin=128, width=299, height=299
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
              x_origin=128, y_origin=0, width=299, height=299
          ),
      ),
  )
  def test_coordinates_conversion(
      self, y_strides, x_strides, patch_size, stride_size, expected_patch_bounds
  ):
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
        patch_size,
        stride_size=stride_size,
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

  def test_patch_generator_with_custom_tissue_mask(self):
    custom_tissue_mask = np.ones((100, 10), dtype=np.bool_)
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    patch_generator = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
        patch_size=256,
        mask=custom_tissue_mask,
    )
    # tissue masks are rescaled, based on patch dim
    self.assertEqual(patch_generator.total_num_patches(), 1000)
    self.assertEqual(patch_generator.stride_width, 9779)
    self.assertEqual(patch_generator.stride_height, 2211)

  def test_patch_generator_with_custom_tissue_mask_raises_invalid_tissue_mask(
      self,
  ):
    custom_tissue_mask = np.ones((3, 3), dtype=np.float32)
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mock_image = mock.create_autospec(dicom_slide.DicomImage)
    mock_slide.get_image.return_value = mock_image
    mock_image.image_bytes.return_value = patch_arr
    with self.assertRaises(ez_wsi_errors.InvalidTissueMaskError):
      patch_generator_lib.DicomPatchGenerator(
          mock_slide,
          _get_mock_slide_level(
              pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
              patch_arr,
              40.0 / 1.25,
          ),
          stride_size=512,
          patch_size=256,
          mask=custom_tissue_mask,
      )

  def test_patch_generator_constructor_with_custom_tissue(self):
    patch_arr = np.array(
        Image.open(dicom_test_utils.testdata_path('low_res_slide_img.png'))
    )
    mock_slide = mock.create_autospec(dicom_slide.DicomSlide)
    custom_tissue_mask = np.ones((2, 2), dtype=np.bool_)
    custom_tissue_mask[0, :] = False
    patch_gen = patch_generator_lib.DicomPatchGenerator(
        mock_slide,
        _get_mock_slide_level(
            pixel_spacing.PixelSpacing.FromMagnificationString('40X'),
            patch_arr,
            40.0 / 1.25,
        ),
        stride_size=512,
        patch_size=256,
        mask=custom_tissue_mask,
    )
    self.assertEqual(
        patch_gen._user_provided_tissue_mask.tolist(), [[0, 0], [255, 255]]
    )

  def test_image_patch_generator(self):
    gen = patch_generator_lib.GcsImagePatchGenerator(
        local_image.LocalImage(dicom_test_utils.test_jpeg_path()),
        stride_size=10,
        patch_size=10,
    )
    self.assertLen(list(gen), 195)

  def test_image_patch_generator_get_index(self):
    gen = patch_generator_lib.GcsImagePatchGenerator(
        local_image.LocalImage(dicom_test_utils.test_jpeg_path()),
        stride_size=10,
        patch_size=10,
    )
    self.assertEqual((gen[1].x, gen[1].y), (50, 10))

  def test_image_patch_generator_get_slice_index(self):
    gen = patch_generator_lib.GcsImagePatchGenerator(
        local_image.LocalImage(dicom_test_utils.test_jpeg_path()),
        stride_size=10,
        patch_size=10,
    )
    self.assertIsInstance(gen[:1], list)
    self.assertLen(gen[:2], 2)
    for item in gen[:1]:
      self.assertIsInstance(item, gcs_image.GcsPatch)

  def test_patch_generator_raises_if_patch_size_exceeds_image_dim(self):
    with self.assertRaises(ez_wsi_errors.InvalidPatchDimensionError):
      gen = patch_generator_lib.GcsImagePatchGenerator(
          local_image.LocalImage(dicom_test_utils.test_jpeg_path()),
          stride_size=10,
          patch_size=99999999,
          mask=np.ones((10, 10), dtype=np.bool_),
      )
      list(gen)

  def test_list_local_images_to_patches(self):
    images = list(
        patch_generator_lib.local_images_to_patches([
            dicom_test_utils.test_jpeg_path(),
            dicom_test_utils.test_jpeg_path(),
        ])
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_iterator_local_images_to_patches(self):
    images = list(
        patch_generator_lib.local_images_to_patches(
            iter([
                dicom_test_utils.test_jpeg_path(),
                dicom_test_utils.test_jpeg_path(),
            ])
        )
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_local_image_to_patches(self):
    images = list(
        patch_generator_lib.local_images_to_patches(
            dicom_test_utils.test_jpeg_path()
        )
    )
    self.assertLen(images, 1)
    for img in images:
      self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_local_image_input_to_patches(self):
    images = list(
        patch_generator_lib.local_images_to_patches(
            local_image.LocalImage(dicom_test_utils.test_jpeg_path())
        )
    )
    self.assertLen(images, 1)
    for img in images:
      self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_gcs_image_single_file_patch_generator(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as _:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        images = list(
            patch_generator_lib.gcs_images_to_patches(
                'gs://test_bucket/test_image.jpg'
            )
        )
        self.assertLen(images, 1)
        for img in images:
          self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_gcs_image_list_file_patch_generator(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as _:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        images = list(
            patch_generator_lib.gcs_images_to_patches([
                'gs://test_bucket/test_image.jpg',
                'gs://test_bucket/test_image.jpg',
            ])
        )
        self.assertLen(images, 2)
        for img in images:
          self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_gcs_image_single_gs_image_patch_generator(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as _:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        images = list(
            patch_generator_lib.gcs_images_to_patches(
                gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
            )
        )
        self.assertLen(images, 1)
        for img in images:
          self.assertIsInstance(img, gcs_image.GcsPatch)

  def test_pixel_spacing_to_level_exact(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      level = patch_generator_lib._pixel_spacing_to_level(
          ds, ds.native_level.pixel_spacing, 0
      )
      self.assertEqual((level.width, level.height), (1152, 700))

  def test_pixel_spacing_to_level_downsample_downsample_not_found_raises(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      ps = pixel_spacing.PixelSpacing(
          ds.native_level.pixel_spacing.column_spacing_mm * 8.0,
          ds.native_level.pixel_spacing.row_spacing_mm * 8.0,
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.PixelSpacingLevelNotFoundError,
          'No pyramid level found with pixel spacing.*',
      ):
        patch_generator_lib._pixel_spacing_to_level(ds, ps, 7.0)

  def test_pixel_spacing_to_level_downsample_downsample_found(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      ps = pixel_spacing.PixelSpacing(
          ds.native_level.pixel_spacing.column_spacing_mm * 8.0,
          ds.native_level.pixel_spacing.row_spacing_mm * 8.0,
      )
      level = patch_generator_lib._pixel_spacing_to_level(ds, ps, 8.0)
      self.assertEqual((level.width, level.height), (1152, 700))

  def test_patch_generator_raises_if_mask_dim_exceed_tissue_dim(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.InvalidTissueMaskError,
          'Tissue mask dimensions exceed image dimensions.',
      ):
        patch_generator_lib.DicomPatchGenerator(
            ds,
            ds.native_level,
            224,
            np.zeros(
                (ds.native_level.width + 1, ds.native_level.height + 1),
                dtype=bool,
            ),
        )

  def test_patch_generator_inits_stride_size_if_undefined(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      gen = patch_generator_lib.DicomPatchGenerator(
          ds,
          ds.native_level,
          224,
      )
      self.assertEqual(gen.stride_width, 224)
      self.assertEqual(gen.stride_height, 224)

  def test_patch_generator_raises_if_mask_incorrectly_init(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      with self.assertRaisesRegex(
          ValueError,
          'Parameter order for patch generator has changed.',
      ):
        patch_generator_lib.DicomPatchGenerator(
            ds,
            ds.native_level,
            224,
            typing.cast(np.ndarray, 225),
        )

  @parameterized.parameters([1, 3])
  def test_normalized_tissue_mask_tissue_mask_shape_input_shape_3_3_dim_(
      self, dim
  ):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      with mock.patch.object(
          patch_generator_lib.DicomPatchGenerator,
          'get_tissue_mask',
          autospec=True,
      ) as mock_get_tissue_mask:
        mock_get_tissue_mask.return_value = (
            mock_get_tissue_mask.return_value
        ) = np.full((10, 10, dim), 100, dtype=np.uint8)
        ds = dicom_slide.DicomSlide(
            dicom_web_interface.DicomWebInterface(
                credential_factory.CredentialFactory()
            ),
            series_path,
        )
        gen = patch_generator_lib.DicomPatchGenerator(ds, ds.native_level, 224)
        expected_mask_shape = (
            int(ds.native_level.height / 224),  # 224 = stride_height
            int(ds.native_level.width / 224),  # 224 = stride_width
        )
        self.assertEqual(
            gen._normalized_tissue_mask().shape, expected_mask_shape
        )

  def test_gen_normalized_tissue_mask_tissue_mask_shape_not_equal_stide_raises(
      self,
  ):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      gen = patch_generator_lib.DicomPatchGenerator(ds, ds.native_level, 224)
      gen.stride_width = 10
      gen.stride_height = 12
      with mock.patch.object(
          patch_generator_lib._BasePatchGenerator,
          'get_tissue_mask',
          autospec=True,
      ) as mock_get_tissue_mask:
        mock_get_tissue_mask.return_value = (
            mock_get_tissue_mask.return_value
        ) = np.zeros((10, 10, 1), dtype=np.uint8)
        with self.assertRaisesRegex(
            ValueError,
            'Stride dimensions must be equal along both dimensions if not'
            ' initialized from user provided mask.',
        ):
          gen._normalized_tissue_mask()

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_dicom_microscope_image_patch_generator_raises_if_init_from_pixel_spacing(
      self, sop_class_uid
  ):
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )

    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      ps = pixel_spacing.PixelSpacing.FromMagnificationString('1.250X')
      with self.assertRaisesRegex(
          ez_wsi_errors.InvalidTissueMaskError,
          'Can not initialize patch generator for DICOM microscopy images'
          ' using pixel spacing to defined the source imaging.',
      ):
        patch_generator_lib.DicomPatchGenerator(ds, ps, 224)

  def test_dicom_generator_raises_if_mask_and_mask_level_undefined(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      gen = patch_generator_lib.DicomPatchGenerator(ds, ds.native_level, 224)
      gen._tissue_mask_level = None
      with self.assertRaisesRegex(ValueError, 'Unexpected object.'):
        gen.get_tissue_mask()

  def test_dicom_generator_init_from_one_frame_dcm_uses_source_as_tissue_level(
      self,
  ):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'
    test_instance.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
    test_instance.ImagedVolumeHeight = 10.0
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.50'
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      gen = patch_generator_lib.DicomPatchGenerator(ds, ds.native_level, 224)
      self.assertEqual(gen._tissue_mask_level, ds.native_level)
      self.assertEqual(
          gen.get_tissue_mask().shape,
          (ds.native_level.height, ds.native_level.width, 3),
      )

  def test_dicom_generator_init_from_resize_one_frame_dcm_uses_source_as_tissue_level(
      self,
  ):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'
    test_instance.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
    test_instance.ImagedVolumeHeight = 10.0
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.50'
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      resize_source = ds.native_level.resize(
          dicom_slide.ImageDimensions(100, 100)
      )
      gen = patch_generator_lib.DicomPatchGenerator(ds, resize_source, 224)
      self.assertEqual(gen._tissue_mask_level, resize_source)
      self.assertEqual(
          gen.get_tissue_mask().shape,
          (resize_source.height, resize_source.width, 3),
      )

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_dicom_microscope_image_mask_set_to_source_image(self, sop_class_uid):
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )

    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mock_store:
      mock_store[store_path].add_instance(test_instance)
      ds = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          series_path,
      )
      first_image = list(ds.levels)[0]
      gen = patch_generator_lib.DicomPatchGenerator(ds, first_image, 224)
      self.assertEqual(gen._tissue_mask_level, first_image)
      self.assertEqual(
          gen.get_tissue_mask().shape,
          (first_image.height, first_image.width, 1),
      )


if __name__ == '__main__':
  absltest.main()
